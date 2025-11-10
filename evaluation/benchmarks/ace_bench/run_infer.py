"""ACE-Bench Inference Script.

This script runs OpenHands agents on ACE-Bench instances and generates patches.
Based on SWE-Bench's run_infer.py implementation.
"""

import asyncio
import copy
import logging
import os
import tempfile
from datetime import datetime
from typing import Any

import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

import openhands.agenthub
from evaluation.benchmarks.ace_bench.binary_patch_utils import (
    remove_binary_diffs,
    remove_binary_files_from_git,
)
from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    check_maximum_retries_exceeded,
    codeact_user_response,
    get_default_sandbox_config_for_eval,
    get_metrics,
    get_openhands_config_for_eval,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
    update_llm_config_for_completions_logging,
)
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig,
    get_agent_config_arg,
    get_evaluation_parser,
    get_llm_config_arg,
    get_llms_for_routing_config,
    get_model_routing_config_arg,
)
from openhands.core.config.condenser_config import NoOpCondenserConfig
from openhands.core.config.utils import get_condenser_config_arg
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.critic import AgentFinishedCritic
from openhands.events.action import Action, CmdRunAction, FileReadAction, MessageAction
from openhands.events.action.agent import AgentFinishAction
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
)
from openhands.events.serialization.event import event_from_dict, event_to_dict
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.shutdown_listener import sleep_if_should_continue

# Environment variables
USE_HINT_TEXT = os.environ.get('USE_HINT_TEXT', 'false').lower() == 'true'
RUN_WITH_BROWSING = os.environ.get('RUN_WITH_BROWSING', 'false').lower() == 'true'
ENABLE_LLM_EDITOR = os.environ.get('ENABLE_LLM_EDITOR', 'false').lower() == 'true'

# Dataset type tracking
DATASET_TYPE = 'ACE-Bench'


def ace_bench_user_response(state: State) -> str:
    """
    ACE-Bench 的 fake user response 函数。

    当检测到 agent 使用 finish 动作时，返回 '/exit' 来结束对话。
    否则返回标准的继续工作消息。
    """
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'When you think you have solved the question, please use the finish tool and include your final answer in the message parameter of the finish tool.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # 检查最后一个 action 是否是 finish
        last_action = next(
            (
                event
                for event in reversed(state.history)
                if isinstance(event, Action)
            ),
            None,
        )

        # 如果最后一个 action 是 AgentFinishAction，返回 /exit 结束对话
        if isinstance(last_action, AgentFinishAction):
            logger.info('Detected AgentFinishAction, ending the conversation.')
            return '/exit'

        # 检查 agent 是否尝试与用户对话超过 2 次
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # 让 agent 知道可以放弃
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


AGENT_CLS_TO_FAKE_USER_RESPONSE_FN = {
    'CodeActAgent': ace_bench_user_response,
}


def get_instruction(
    instance: pd.Series,
    metadata: EvalMetadata,
    runtime: Runtime = None,
) -> MessageAction:
    """Generate instruction for the agent based on the instance.

    ACE-Bench 的指令从模板文件生成，直接包含问题描述

    Args:
        instance: ACE-Bench instance data
        metadata: Evaluation metadata
        runtime: Runtime instance (保留参数以兼容旧代码，但不再使用)

    Returns:
        MessageAction containing the instruction
    """
    # 生成系统提示
    # Determine template file
    if metadata.instruction_template_name:
        template_name = metadata.instruction_template_name
    else:
        template_name = 'ace_default.j2'

    logger.debug(f'Using instruction template file: {template_name}')

    # Set up Jinja2 environment
    prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
    env = Environment(loader=FileSystemLoader(prompts_dir))
    template = env.get_template(template_name)

    # Prepare context for rendering (包含任务描述)
    context = {
        'instance': instance,
        'metadata': metadata,
        'task_prompt': instance.get('problem_statement', ''),
    }

    # Render the instruction
    instruction = template.render(context)

    if RUN_WITH_BROWSING:
        instruction += (
            '<IMPORTANT!>\nYou SHOULD NEVER attempt to browse the web. </IMPORTANT!>\n'
        )

    return MessageAction(content=instruction)


def get_instance_docker_image(instance: pd.Series) -> str:
    """Get Docker image name for the instance.

    Args:
        instance: ACE-Bench instance data

    Returns:
        Docker image name (格式: docker.io/{image_name})
    """
    # ACE-Bench 数据集中的 image_name 字段直接包含完整的镜像名
    if 'image_name' in instance and pd.notna(instance['image_name']):
        image_name = instance['image_name']
        # 如果没有包含registry前缀，添加docker.io前缀
        if '/' not in image_name or not image_name.startswith(('docker.io/', 'gcr.io/', 'ghcr.io/')):
            return f'docker.io/{image_name}'.lower()
        return image_name.lower()

    # 如果没有指定 image_name，抛出错误
    raise ValueError(f'image_name not found for instance {instance.instance_id}')


def get_config(
    instance: pd.Series,
    metadata: EvalMetadata,
) -> OpenHandsConfig:
    """Get OpenHands configuration for the instance.

    Args:
        instance: ACE-Bench instance data
        metadata: Evaluation metadata

    Returns:
        OpenHands configuration
    """
    base_container_image = get_instance_docker_image(instance)
    logger.info(
        f'Using instance container image: {base_container_image}. '
        f'Please make sure this image exists. '
        f'Submit an issue on https://github.com/OpenHands/OpenHands if you run into any issues.'
    )

    sandbox_config = get_default_sandbox_config_for_eval()
    sandbox_config.base_container_image = base_container_image
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False
    sandbox_config.platform = 'linux/amd64'

    # TODO: 如果需要动态调整资源，可以在此处设置
    # sandbox_config.remote_runtime_resource_factor = get_resource_factor(instance)

    config = get_openhands_config_for_eval(
        metadata=metadata,
        enable_browser=RUN_WITH_BROWSING,
        runtime=os.environ.get('RUNTIME', 'docker'),
        sandbox_config=sandbox_config,
    )

    config.set_llm_config(
        update_llm_config_for_completions_logging(
            metadata.llm_config, metadata.eval_output_dir, instance['instance_id']
        )
    )
    config.set_llm_config(get_llm_config_arg('draft_editor'), 'draft_editor')

    model_routing_config = get_model_routing_config_arg()
    model_routing_config.llms_for_routing = get_llms_for_routing_config()

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=RUN_WITH_BROWSING,
        enable_llm_editor=ENABLE_LLM_EDITOR,
        enable_mcp=False,
        condenser=metadata.condenser_config,
        enable_prompt_extensions=False,
        model_routing=model_routing_config,
        system_prompt_filename=metadata.agent_config.system_prompt_filename
        if metadata.agent_config
        else 'system_prompt.j2',
    )
    config.set_agent_config(agent_config)

    return config


def initialize_runtime(
    runtime: Runtime,
    instance: pd.Series,
    metadata: EvalMetadata,
):
    """Initialize the runtime for the agent.

    This function sets up the container environment before agent execution.

    ACE-Bench 特定的初始化流程：
    1. 根据 level 进行不同的预处理：
       - Level 1: 激活环境、复制仓库、apply patch 和 test_patch、初始化 git
       - Level 2: 清空 /testbed/、创建新仓库、初始化 git

    Args:
        runtime: Runtime instance
        instance: ACE-Bench instance data
        metadata: Evaluation metadata
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Initialization Fn')
    logger.info('-' * 30)

    obs: CmdOutputObservation

    # 设置 instance id 和基本配置
    action = CmdRunAction(
        command=f"""echo 'export ACE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && """
        f"""echo 'export PIP_CACHE_DIR=~/.cache/pip' >> ~/.bashrc && """
        f"""echo "alias git='git --no-pager'" >> ~/.bashrc && """
        f"""git config --global core.pager "" && """
        f"""git config --global diff.binary false"""
    )
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        obs.exit_code == 0,
        f'Failed to export ACE_INSTANCE_ID and configure git: {str(obs)}',
    )

    # 导出 USER 变量
    action = CmdRunAction(command='export USER=$(whoami); echo USER=${USER}')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to export USER: {str(obs)}')

    # 根据 level 进行不同的处理
    instance_level = int(instance['level'])
    logger.info(f'Instance level: {instance_level}')

    if instance_level == 1:
        # Level 1 处理
        logger.info('Processing Level 1 instance...')

        # Step 1: 激活环境并复制仓库
        logger.info('Step 1: Activating conda environment and restoring project')
        conda_activate_cmd = 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed'
        restore_project_cmd = 'rm -rf /testbed/* && cp -r /root/my_repo/* /testbed/'

        action = CmdRunAction(command=f'{conda_activate_cmd} && {restore_project_cmd}')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to activate conda and restore project: {str(obs)}',
        )

        # Step 2: Apply patch 到 /testbed/ (用于 mask 文件)
        logger.info('Step 2: Applying patch to mask files in /testbed/')
        patch_content = instance.get('patch', '')

        if patch_content and patch_content.strip():
            # 创建临时 patch 文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False, encoding='utf-8') as f:
                f.write(patch_content)
                temp_patch_path = f.name

            try:
                # 复制 patch 文件到容器
                runtime.copy_to(temp_patch_path, '/tmp/')
                patch_filename = os.path.basename(temp_patch_path)
                container_patch_path = f'/tmp/{patch_filename}'

                # Apply patch
                action = CmdRunAction(
                    command=f'cd /testbed && git apply --whitespace=fix {container_patch_path}'
                )
                action.set_hard_timeout(600)
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                if obs.exit_code != 0:
                    logger.warning(f'Failed to apply patch: {obs.content}')
                else:
                    logger.info('Successfully applied patch for masking')
            finally:
                # 清理临时文件
                os.unlink(temp_patch_path)
        else:
            logger.info('No patch to apply for masking')

        # Step 3: 直接删除 F2P 测试文件（不使用 test_patch）
        logger.info('Step 3: Deleting F2P test files')

        # 从 instance 中获取 FAIL_TO_PASS 测试路径
        fail_to_pass = instance.get('FAIL_TO_PASS', [])
        if fail_to_pass:
            f2p_tests = fail_to_pass if isinstance(fail_to_pass, list) else [fail_to_pass]

            for f2p_test in f2p_tests:
                # 确保路径以 /testbed/ 开头
                if not f2p_test.startswith('/testbed/'):
                    f2p_test_path = f'/testbed/{f2p_test}'
                else:
                    f2p_test_path = f2p_test

                logger.info(f'Deleting F2P test file: {f2p_test_path}')

                # 删除测试文件
                action = CmdRunAction(command=f'rm -f {f2p_test_path}')
                action.set_hard_timeout(600)
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                if obs.exit_code != 0:
                    logger.warning(f'Failed to delete F2P test file {f2p_test_path}: {obs.content}')
                else:
                    logger.info(f'Successfully deleted F2P test file: {f2p_test_path}')
        else:
            logger.warning('No FAIL_TO_PASS tests found in instance, skipping test file deletion')

        # Step 4: 重新初始化 /testbed/ 下的 git 仓库
        logger.info('Step 4: Re-initializing git repository in /testbed/')

        # 删除现有的 .git 目录
        action = CmdRunAction(command='cd /testbed && rm -rf .git')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to remove .git in /testbed: {str(obs)}',
        )

        # 初始化新的 git 仓库
        action = CmdRunAction(command='cd /testbed && git init')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to git init in /testbed: {str(obs)}',
        )

        # 配置 git 用户信息（如果需要）
        action = CmdRunAction(
            command='cd /testbed && git config user.email "ace@bench.com" && git config user.name "ACE Bench"'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to configure git user: {str(obs)}',
        )

        # 进行第一次提交
        action = CmdRunAction(
            command='cd /testbed && git add -A && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create initial commit in /testbed: {str(obs)}',
        )

        # 获取并记录初始 commit hash
        action = CmdRunAction(command='cd /testbed && git rev-parse HEAD')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to get initial commit hash: {str(obs)}',
        )
        initial_commit = obs.content.strip()
        logger.info(f'Initial commit hash: {initial_commit}')

    elif instance_level == 2:
        # Level 2 处理
        logger.info('Processing Level 2 instance...')

        # 2.1) 激活 conda 环境
        logger.info('Step 2.1: Activating conda environment')
        conda_activate_cmd = 'source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed'
        action = CmdRunAction(command=conda_activate_cmd)
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to activate conda environment: {str(obs)}',
        )

        # 2.2) 删除 /testbed/ 下所有内容
        logger.info('Step 2.2: Cleaning /testbed/ directory')
        action = CmdRunAction(command='rm -rf /testbed/* /testbed/.*  2>/dev/null || true')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        # 使用 || true 确保命令总是成功

        # 确保 /testbed 目录存在
        action = CmdRunAction(command='mkdir -p /testbed')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create /testbed directory: {str(obs)}',
        )

        # 初始化 git 仓库
        action = CmdRunAction(command='cd /testbed && git init')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to git init in /testbed: {str(obs)}',
        )

        # 配置 git 用户信息
        action = CmdRunAction(
            command='cd /testbed && git config user.email "ace@bench.com" && git config user.name "ACE Bench"'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to configure git user: {str(obs)}',
        )

        # 创建 README.md
        action = CmdRunAction(
            command='cd /testbed && echo "put all codes in this folder" > README.md'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create README.md: {str(obs)}',
        )

        # 进行第一次提交
        action = CmdRunAction(
            command='cd /testbed && git add -A && git commit -m "Initial commit for ACE-Bench evaluation" --allow-empty'
        )
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to create initial commit in /testbed: {str(obs)}',
        )

        # 获取并记录初始 commit hash
        action = CmdRunAction(command='cd /testbed && git rev-parse HEAD')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert_and_raise(
            obs.exit_code == 0,
            f'Failed to get initial commit hash: {str(obs)}',
        )
        initial_commit = obs.content.strip()
        logger.info(f'Initial commit hash: {initial_commit}')

    logger.info('-' * 30)
    logger.info('END Runtime Initialization Fn')
    logger.info('-' * 30)


def complete_runtime(
    runtime: Runtime,
    instance: pd.Series,
) -> dict[str, Any]:
    """Complete the runtime and extract results.

    This function is called after the agent finishes to extract the patch.

    ACE-Bench 的处理流程：
    1. 切换到 /testbed/ 目录（agent 的工作目录）
    2. 添加所有文件到 git
    3. 与初始 commit 进行 diff 生成 patch
    4. 返回 patch（格式与 SWE-Bench 对齐）

    Args:
        runtime: Runtime instance
        instance: ACE-Bench instance data

    Returns:
        Dictionary containing the git patch
    """
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)

    obs: CmdOutputObservation

    # 切换到 /testbed/ 目录（agent 工作的代码仓库）
    action = CmdRunAction(command='cd /testbed')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    # 处理可能还在运行的命令
    if obs.exit_code == -1:
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command='cd /testbed')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code == -1:
        logger.info('The previous command is still running, trying to ctrl+z it...')
        action = CmdRunAction(command='C-z')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command='cd /testbed')
        action.set_hard_timeout(600)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to cd to /testbed: {str(obs)}',
    )

    # 配置 git（禁用 pager）
    action = CmdRunAction(command='git config --global core.pager ""')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git config --global core.pager "": {str(obs)}',
    )

    # 检查并移除嵌套的 git 仓库（避免 git 命令出现问题）
    action = CmdRunAction(command='find . -type d -name .git -not -path "./.git"')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to find git repositories: {str(obs)}',
    )

    # 移除嵌套的 git 仓库
    git_dirs = [p for p in obs.content.strip().split('\n') if p]
    if git_dirs:
        for git_dir in git_dirs:
            action = CmdRunAction(command=f'rm -rf "{git_dir}"')
            action.set_hard_timeout(600)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            assert_and_raise(
                isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                f'Failed to remove git directory {git_dir}: {str(obs)}',
            )

    # 添加所有文件到 git staging area
    action = CmdRunAction(command='git add -A')
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to git add -A: {str(obs)}',
    )

    # 从 git staging 中移除二进制文件（参考 SWE-Bench 实现）
    action = CmdRunAction(command=remove_binary_files_from_git())
    action.set_hard_timeout(600)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to remove binary files: {str(obs)}',
    )

    # 生成 git diff（与初始 commit 对比）
    n_retries = 0
    git_patch = None
    while n_retries < 5:
        # 获取初始 commit hash（第一个 commit）
        action = CmdRunAction(command='git rev-list --max-parents=0 HEAD')
        action.set_hard_timeout(300)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            base_commit = obs.content.strip().split('\n')[0]
            logger.info(f'Base commit for diff: {base_commit}')
        else:
            logger.warning('Failed to get base commit, using HEAD as fallback')
            base_commit = 'HEAD'

        # 生成 diff 并保存到文件（使用 SWE-Bench 的方式）
        action = CmdRunAction(
            command=f'git diff --no-color --cached {base_commit} > patch.diff'
        )
        action.set_hard_timeout(max(300 + 100 * n_retries, 600))
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        n_retries += 1

        if isinstance(obs, CmdOutputObservation):
            if obs.exit_code == 0:
                # 读取 patch 文件
                action = FileReadAction(path='patch.diff')
                action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                logger.info(action, extra={'msg_type': 'ACTION'})
                obs = runtime.run_action(action)
                logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                if isinstance(obs, FileReadObservation):
                    git_patch = obs.content
                    logger.info(f'Successfully read patch via FileReadAction ({len(git_patch)} chars)')
                    break
                elif isinstance(obs, ErrorObservation):
                    # Fallback: 使用 cat 命令读取
                    # 与 SWE-Bench 一样，文件可能无法被解码为 utf-8
                    logger.warning(f'FileReadAction failed: {obs.content}. Trying cat command...')
                    action = CmdRunAction(command='cat patch.diff')
                    action.set_hard_timeout(max(300 + 100 * n_retries, 600))
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                    assert_and_raise(
                        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
                        f'Failed to read patch via cat: {str(obs)}'
                    )
                    git_patch = obs.content
                    logger.info(f'Successfully read patch via cat ({len(git_patch)} chars)')
                    break
                else:
                    assert_and_raise(False, f'Unexpected observation type: {str(obs)}')
            else:
                logger.warning(f'Failed to generate git diff (exit code {obs.exit_code}), retrying...')
                sleep_if_should_continue(10)
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error occurred: {obs.content}. Retrying...')
            sleep_if_should_continue(10)
        else:
            assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    assert_and_raise(git_patch is not None, 'Failed to get git diff (None)')

    # 从 patch 中移除二进制 diff 块（参考 SWE-Bench 实现）
    git_patch = remove_binary_diffs(git_patch)

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)

    return {'git_patch': git_patch}


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Process a single ACE-Bench instance.

    Args:
        instance: ACE-Bench instance data
        metadata: Evaluation metadata
        reset_logger: Whether to reset logger for multiprocessing
        runtime_failure_count: Number of runtime failures so far

    Returns:
        Evaluation output
    """
    config = get_config(instance, metadata)

    # Setup logger
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance.instance_id, log_dir)
    else:
        logger.info(f'Starting evaluation for instance {instance.instance_id}.')

    # Increase resource factor with increasing attempt
    if runtime_failure_count > 0:
        config.sandbox.remote_runtime_resource_factor = min(
            config.sandbox.remote_runtime_resource_factor * (2**runtime_failure_count),
            8,
        )
        logger.warning(
            f'This is the {runtime_failure_count + 1}th attempt for instance {instance.instance_id}, '
            f'setting resource factor to {config.sandbox.remote_runtime_resource_factor}'
        )

    metadata = copy.deepcopy(metadata)
    metadata.details['runtime_failure_count'] = runtime_failure_count
    metadata.details['remote_runtime_resource_factor'] = (
        config.sandbox.remote_runtime_resource_factor
    )

    runtime = create_runtime(config)
    call_async_from_sync(runtime.connect)

    try:
        # 初始化运行时环境（包括挂载任务文件、处理 masked files、初始化 git 等）
        initialize_runtime(runtime, instance, metadata)

        # 获取指令（从容器内读取任务描述）
        message_action = get_instruction(instance, metadata, runtime=runtime)

        # 运行 agent
        state: State | None = asyncio.run(
            run_controller(
                config=config,
                initial_user_action=message_action,
                runtime=runtime,
                fake_user_response_fn=AGENT_CLS_TO_FAKE_USER_RESPONSE_FN[
                    metadata.agent_class
                ],
            )
        )

        # Check for fatal errors
        if is_fatal_evaluation_error(state.last_error):
            raise EvalException('Fatal error detected: ' + state.last_error)

        # Extract results
        return_val = complete_runtime(runtime, instance)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {instance.instance_id}:\n--------\n{git_patch}\n--------'
        )
    finally:
        runtime.close()

    # Prepare test result
    test_result = {
        'git_patch': git_patch,
    }

    if state is None:
        raise ValueError('State should not be None.')

    # Get histories and metrics
    histories = [event_to_dict(event) for event in state.history]
    metrics = get_metrics(state)

    # Save the output
    instruction = message_action.content
    if message_action.image_urls:
        instruction += (
            '\n\n<image_urls>' + '\n'.join(message_action.image_urls) + '</image_urls>'
        )

    output = EvalOutput(
        instance_id=instance.instance_id,
        instruction=instruction,
        instance=instance.to_dict(),
        test_result=test_result,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=state.last_error if state and state.last_error else None,
    )
    return output


def filter_dataset(
    dataset: pd.DataFrame,
    filter_column: str,
    selected_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Filter dataset based on selected_ids, config.toml or environment variables.

    Args:
        dataset: Full dataset
        filter_column: Column name to filter on (e.g., 'instance_id')
        selected_ids: List of specific IDs to evaluate (from command line argument)

    Returns:
        Filtered dataset
    """
    # 优先使用命令行参数指定的 selected_ids
    if selected_ids is not None and len(selected_ids) > 0:
        logger.info(
            f'Filtering {len(selected_ids)} tasks from --selected-ids argument...'
        )
        subset = dataset[dataset[filter_column].isin(selected_ids)]
        logger.info(f'Retained {subset.shape[0]} tasks after filtering')
        return subset

    # 其次尝试从 config.toml 读取
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            # 支持通过 selected_ids 指定要评估的特定实例
            if 'selected_ids' in data:
                selected_ids_from_config = data['selected_ids']
                logger.info(
                    f'Filtering {len(selected_ids_from_config)} tasks from config.toml "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids_from_config)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset

    # 支持通过环境变量 SKIP_IDS 跳过特定实例
    skip_ids = os.environ.get('SKIP_IDS', '').split(',')
    if len(skip_ids) > 0 and skip_ids[0]:
        logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
        return dataset[~dataset[filter_column].isin(skip_ids)]

    return dataset


if __name__ == '__main__':

    # 在模块加载时应用 patch
    from utils import patch_llm_completion_logging
    patch_llm_completion_logging()

    parser = get_evaluation_parser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='ace-bench',
        help='Dataset name (for ACE-Bench, this is just a label)',
    )
    parser.add_argument(
        '--level',
        type=str,
        default='1,2',
        help='Comma-separated list of levels to evaluate (e.g., "1", "2", or "1,2" for both)',
    )
    parser.add_argument(
        '--selected-ids',
        type=str,
        nargs='+',
        default=None,
        help='Specific instance IDs to evaluate (space-separated)',
    )

    args, _ = parser.parse_known_args()

    # 解析 level 参数
    levels_to_eval = [int(l.strip()) for l in args.level.split(',') if l.strip()]
    logger.info(f'Evaluating levels: {levels_to_eval}')

    # ACE-Bench: 从 HuggingFace 加载数据
    logger.info('Loading ACE-Bench dataset from HuggingFace...')

    all_instances = []

    # 获取 HuggingFace token（如果是私有数据集）
    hf_token = os.environ.get('HF_TOKEN', None)
    if hf_token:
        logger.info('Using HuggingFace token from HF_TOKEN environment variable')
    else:
        logger.warning('HF_TOKEN not found. If this is a private dataset, please set it.')

    # 清理可能损坏的缓存
    import shutil
    cache_dir = os.path.expanduser('~/.cache/huggingface/datasets/BamChil___ace-bench')
    if os.path.exists(cache_dir):
        logger.info(f'Found existing cache at {cache_dir}, removing to avoid corruption issues...')
        try:
            shutil.rmtree(cache_dir)
            logger.info('Cache cleared successfully')
        except Exception as e:
            logger.warning(f'Failed to clear cache: {e}')

    try:
        # 根据指定的 level 加载数据
        if 1 in levels_to_eval:
            logger.info('Loading Level 1 tasks from HuggingFace...')
            dataset_lv1 = load_dataset(
                "BamChil/ACE-Bench",
                split="level1",
                token=hf_token,
            )
            # 转换为 pandas DataFrame
            df_lv1 = pd.DataFrame(dataset_lv1)
            df_lv1['level'] = 1
            all_instances.append(df_lv1)
            logger.info(f'Loaded {len(df_lv1)} Level 1 instances')

        if 2 in levels_to_eval:
            logger.info('Loading Level 2 tasks from HuggingFace...')
            dataset_lv2 = load_dataset(
                "BamChil/ACE-Bench",
                split="level2",
                token=hf_token,
            )
            # 转换为 pandas DataFrame
            df_lv2 = pd.DataFrame(dataset_lv2)
            df_lv2['level'] = 2
            all_instances.append(df_lv2)
            logger.info(f'Loaded {len(df_lv2)} Level 2 instances')

        # 合并所有实例
        if not all_instances:
            raise ValueError(f'No valid levels specified. Please specify levels from: 1, 2')

        ace_bench_tests = pd.concat(all_instances, ignore_index=True)
        logger.info(f'Total loaded: {len(ace_bench_tests)} instances')
    except Exception as e:
        logger.error(f'Failed to load dataset from HuggingFace: {str(e)}')
        logger.error('Troubleshooting steps:')
        logger.error('1. If this is a private dataset, set HF_TOKEN: export HF_TOKEN=your_token')
        logger.error('2. Try manually clearing cache: rm -rf ~/.cache/huggingface/datasets/BamChil___ace-bench')
        logger.error('3. Check if you have access to the dataset at: https://huggingface.co/datasets/BamChil/ACE-Bench')
        raise

    # 过滤数据集（通过命令行参数、配置文件或环境变量）
    ace_bench_tests = filter_dataset(ace_bench_tests, 'instance_id', selected_ids=args.selected_ids)
    logger.info(f'After filtering: {len(ace_bench_tests)} tasks to evaluate')

    # Get LLM config
    llm_config = None
    if args.llm_config:
        llm_config = get_llm_config_arg(args.llm_config, args.config_file)
        llm_config.log_completions = True
        llm_config.modify_params = False

    if llm_config is None:
        raise ValueError(f'Could not find LLM config: --llm_config {args.llm_config}')

    # Get condenser config
    condenser_name = os.environ.get('EVAL_CONDENSER')
    if condenser_name:
        condenser_config = get_condenser_config_arg(condenser_name, args.config_file)
        if condenser_config is None:
            raise ValueError(
                f'Could not find Condenser config: EVAL_CONDENSER={condenser_name}'
            )
    else:
        condenser_config = NoOpCondenserConfig()
        logger.debug(
            'No Condenser config provided via EVAL_CONDENSER, using NoOpCondenser.'
        )

    # Get agent config
    agent_config = None
    if args.agent_config:
        agent_config = get_agent_config_arg(args.agent_config, args.config_file)

    details = {}
    _agent_cls = openhands.agenthub.Agent.get_cls(args.agent_cls)

    # 设置 dataset description
    dataset_description = 'ace-bench'
    metadata = make_metadata(
        llm_config,
        dataset_description,
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
        details=details,
        agent_config=agent_config,
        condenser_config=condenser_config,
    )

    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')
    print(f'### OUTPUT FILE: {output_file} ###')

    # 设置终端输出保存到日志文件
    logs_dir = os.path.join(metadata.eval_output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    terminal_log_file = os.path.join(logs_dir, f'terminal_output_{timestamp}.log')

    # 添加文件 handler 到 logger
    file_handler = logging.FileHandler(terminal_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)

    # 添加到 root logger 和 openhands logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.addHandler(file_handler)

    logger.info(f'Terminal output will be saved to: {terminal_log_file}')
    print(f'### TERMINAL LOG FILE: {terminal_log_file} ###')

    # Prepare dataset
    instances = prepare_dataset(ace_bench_tests, output_file, args.eval_n_limit)

    # Run evaluation
    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=8 * 60 * 60,  # 8 hours per instance
        max_retries=5,
    )

    # Check for maximum retries
    check_maximum_retries_exceeded(metadata.eval_output_dir)

