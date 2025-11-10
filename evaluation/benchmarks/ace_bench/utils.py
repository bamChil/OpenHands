# ===== 自定义 LLM Completions Logging =====
# 这个 monkey patch 将 LLM completions 记录到一个格式整齐的 JSON 文件中
# 而不是 OpenHands 默认的每次调用生成一个单行 JSON 文件的方式

import os
import json
import time
import threading
from functools import wraps
from openhands.core.logger import openhands_logger as logger

# 用于保护文件写入的锁
_completion_log_lock = threading.Lock()

def serialize_for_json(obj):
    """
    将对象序列化为可 JSON 化的格式，处理 Message、ModelResponse、ChatCompletionMessageToolCall 等特殊对象

    Args:
        obj: 要序列化的对象

    Returns:
        可 JSON 序列化的对象
    """
    if obj is None:
        return None

    # 基本类型直接返回
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # 处理列表
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]

    # 处理字典
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}

    # 获取对象的类名
    class_name = obj.__class__.__name__ if hasattr(obj, '__class__') else None

    # 处理 Message 对象
    if class_name == 'Message':
        result = {
            'role': getattr(obj, 'role', None),
            'content': serialize_for_json(getattr(obj, 'content', None)),
        }
        # 保留 tool_calls 字段（如果存在）
        if hasattr(obj, 'tool_calls'):
            tool_calls = getattr(obj, 'tool_calls', None)
            if tool_calls is not None:
                result['tool_calls'] = serialize_for_json(tool_calls)
        return result

    # 处理 ModelResponse 对象（OpenAI API 响应）
    if class_name in ('ModelResponse', 'ChatCompletion'):
        if hasattr(obj, '__dict__'):
            return serialize_for_json(obj.__dict__)
        # 尝试使用 model_dump 方法（Pydantic v2）
        elif hasattr(obj, 'model_dump'):
            try:
                return obj.model_dump()
            except:
                pass
        # 尝试使用 dict 方法（Pydantic v1）
        elif hasattr(obj, 'dict'):
            try:
                return obj.dict()
            except:
                pass

    # 处理 ChatCompletionMessageToolCall 对象
    if class_name == 'ChatCompletionMessageToolCall' or 'ToolCall' in class_name:
        result = {}
        # 提取所有相关字段
        for attr in ['id', 'type', 'function', 'index']:
            if hasattr(obj, attr):
                val = getattr(obj, attr, None)
                if val is not None:
                    result[attr] = serialize_for_json(val)
        return result if result else serialize_for_json(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)

    # 处理 Function 对象（tool_call 中的 function）
    if class_name == 'Function' or (class_name and 'Function' in class_name):
        result = {}
        for attr in ['name', 'arguments']:
            if hasattr(obj, attr):
                val = getattr(obj, attr, None)
                if val is not None:
                    result[attr] = serialize_for_json(val)
        return result if result else serialize_for_json(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)

    # 处理 Choice 对象
    if class_name == 'Choice' or 'Choice' in class_name:
        result = {}
        for attr in ['index', 'message', 'finish_reason', 'logprobs']:
            if hasattr(obj, attr):
                val = getattr(obj, attr, None)
                if val is not None:
                    result[attr] = serialize_for_json(val)
        return result if result else serialize_for_json(obj.__dict__) if hasattr(obj, '__dict__') else str(obj)

    # 处理其他对象：尝试转换为字典
    if hasattr(obj, '__dict__'):
        try:
            obj_dict = obj.__dict__
            # 过滤掉私有属性和方法
            return serialize_for_json({k: v for k, v in obj_dict.items() if not k.startswith('_')})
        except:
            return str(obj)

    # 尝试 Pydantic 的序列化方法
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except:
            pass
    elif hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except:
            pass

    # 最后尝试转换为字符串
    return str(obj)

def custom_log_completion(log_file, log_entry):
    """
    将 completion 记录到单个格式整齐的 JSON 文件中

    优化的结构：
    - initial_messages: 只记录一次（系统提示和初始任务）
    - tools_definition: 只记录一次（所有可用的工具定义）
    - completions: 每次调用的增量内容，包含：
      - id: 递增的调用序号
      - assistant_message: LLM 的输出内容
      - tool_calls: 如果有工具调用，显示调用的工具名称和参数
      - tool_results: 工具返回的结果（简化版）
      - response_meta: 响应元数据（耗时、finish_reason等）

    Args:
        log_file: JSON 文件路径
        log_entry: 要记录的数据（dict），包含 messages 和 response
    """
    try:
        # 序列化 log_entry，处理 Message 等特殊对象
        serialized_entry = serialize_for_json(log_entry)

        # 使用锁保护文件操作，确保多线程安全
        with _completion_log_lock:
            # 读取或初始化日志结构
            log_data = {
                'initial_messages': None,
                'tools_definition': None,
                'completions': [],
                '_last_message_count': 0  # 用于跟踪上次记录的消息数量
            }

            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        if isinstance(existing_data, dict):
                            log_data = existing_data
                            # 确保有 _last_message_count 字段
                            if '_last_message_count' not in log_data:
                                log_data['_last_message_count'] = len(log_data.get('initial_messages', [])) if log_data.get('initial_messages') else 0
                except (json.JSONDecodeError, IOError):
                    pass

            messages = serialized_entry.get('messages', [])
            kwargs = serialized_entry.get('kwargs', {})
            response = serialized_entry.get('response', {})

            # 如果是第一次调用，保存 initial_messages 和 tools_definition
            if log_data['initial_messages'] is None and messages:
                # 前两条通常是系统消息和初始用户任务
                log_data['initial_messages'] = messages[:2] if len(messages) >= 2 else messages
                log_data['_last_message_count'] = len(log_data['initial_messages'])

                # 保存工具定义（只保存工具名称列表，不保存完整schema）
                if 'tools' in kwargs:
                    log_data['tools_definition'] = [
                        tool.get('function', {}).get('name', 'unknown')
                        for tool in kwargs['tools']
                        if isinstance(tool, dict) and 'function' in tool
                    ]

            # 提取本次新增的消息（从上次记录点之后）
            last_count = log_data['_last_message_count']
            new_messages = messages[last_count:] if len(messages) > last_count else []

            # 从 response 中提取工具调用信息
            assistant_message = None
            tool_calls = []

            # 先从 response 的 choices[0].message 中提取助手消息和工具调用
            if response:
                choices = response.get('choices', [])
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get('message', {})

                        # 提取 assistant 的文本内容
                        content = message.get('content')
                        if content:
                            if isinstance(content, str):
                                assistant_message = content
                            elif isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text_parts.append(item.get('text', ''))
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                if text_parts:
                                    assistant_message = '\n'.join(text_parts)

                        # 提取工具调用信息（从response的message中）
                        msg_tool_calls = message.get('tool_calls', [])
                        if msg_tool_calls and isinstance(msg_tool_calls, list):
                            for tc in msg_tool_calls:
                                if isinstance(tc, dict):
                                    func_info = tc.get('function', {})
                                    if not func_info:
                                        continue

                                    tool_name = func_info.get('name', 'unknown')

                                    # 解析参数（可能是JSON字符串）
                                    arguments = func_info.get('arguments', {})
                                    if isinstance(arguments, str):
                                        try:
                                            arguments = json.loads(arguments)
                                        except:
                                            pass

                                    # 特殊处理某些关键参数，让日志更易读
                                    formatted_args = {}
                                    if isinstance(arguments, dict):
                                        for key, value in arguments.items():
                                            # 对于某些参数，提取关键信息
                                            if key == 'command' and tool_name == 'execute_bash':
                                                # bash命令是关键信息，完整保留
                                                formatted_args[key] = value
                                            elif key == 'path':
                                                formatted_args[key] = value
                                            elif key == 'old_str' or key == 'new_str':
                                                # 对于编辑操作，截断过长的字符串
                                                if isinstance(value, str) and len(value) > 200:
                                                    formatted_args[key] = value[:200] + '... (truncated)'
                                                else:
                                                    formatted_args[key] = value
                                            elif key in ('security_risk', 'is_input', 'timeout', 'thought', 'message', 'command_name', 'view_range'):
                                                formatted_args[key] = value
                                            else:
                                                # 其他参数根据长度决定是否截断
                                                if isinstance(value, str) and len(value) > 100:
                                                    formatted_args[key] = value[:100] + '...'
                                                else:
                                                    formatted_args[key] = value
                                    else:
                                        formatted_args = arguments

                                    tool_calls.append({
                                        'tool': tool_name,
                                        'arguments': formatted_args
                                    })

            # 从新增消息中提取工具返回结果
            tool_results = []
            for msg in new_messages:
                if isinstance(msg, dict):
                    role = msg.get('role')
                    content = msg.get('content', [])

                    if role == 'tool':
                        # 提取工具返回结果（限制长度）
                        result_text = []
                        for item in content if isinstance(content, list) else [content]:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text = item.get('text', '')
                                # 限制每个工具结果的长度
                                if len(text) > 500:
                                    text = text[:500] + '... (truncated)'
                                result_text.append(text)
                            elif isinstance(item, str):
                                text = item[:500] + '... (truncated)' if len(item) > 500 else item
                                result_text.append(text)
                        if result_text:
                            tool_results.append('\n'.join(result_text))

            # 创建紧凑的 completion 条目
            completion_entry = {
                'id': len(log_data['completions']) + 1,
                'timestamp': serialized_entry.get('timestamp'),
                'model': serialized_entry.get('model'),
            }

            # 添加工具结果(这个工具结果是上次工具调用的结果)
            if tool_results:
                completion_entry['results_the_last_tool_call'] = tool_results

            # 添加 assistant 消息
            if assistant_message:
                completion_entry['assistant_message'] = assistant_message

            # 添加工具调用信息
            if tool_calls:
                completion_entry['tool_calls'] = tool_calls

            # 添加响应元数据
            if response:
                choices = response.get('choices', [])
                if choices and isinstance(choices, list):
                    first_choice = choices[0]
                    completion_entry['response_meta'] = {
                        'finish_reason': first_choice.get('finish_reason'),
                        'response_ms': response.get('_response_ms'),
                    }

            # 更新消息计数
            log_data['_last_message_count'] = len(messages)

            # 追加新的 completion
            log_data['completions'].append(completion_entry)

            # 写回文件，使用格式整齐的 JSON（indent=2）
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # 如果 logging 失败，记录错误但不影响主流程
        logger.warning(f'Failed to log LLM completion: {e}')

def patch_llm_completion_logging():
    """
    Monkey patch OpenHands 的 LLM 类，将 completions logging 改为单文件格式整齐的方式。

    原始行为：每次 LLM 调用创建一个新的 JSON 文件，内容是单行
    修改后行为：所有调用记录到一个 JSON 文件中，格式整齐（使用 indent），作为 JSON 数组追加
    """
    from openhands.llm.llm import LLM

    # 保存原始的 __init__ 方法
    original_init = LLM.__init__

    def patched_init(self, config, service_id, metrics=None, retry_listener=None):
        """修改后的 __init__，使用自定义的 completion logging"""
        # 保存原始的 log_completions 设置
        original_log_completions = config.log_completions
        original_log_folder = config.log_completions_folder

        # 临时禁用 log_completions，让原始的 __init__ 不创建 logging 逻辑
        config.log_completions = False

        # 调用原始的 __init__
        original_init(self, config, service_id, metrics, retry_listener)

        # 恢复配置
        self.config.log_completions = original_log_completions

        # 如果启用了 log_completions，添加我们自定义的 logging wrapper
        if self.config.log_completions and original_log_folder:
            self.config.log_completions_folder = original_log_folder
            os.makedirs(self.config.log_completions_folder, exist_ok=True)

            # 保存原始的 _completion（此时已经被 wrapper 包装过，但不包含 logging）
            original_completion = self._completion

            @wraps(original_completion)
            def custom_logging_wrapper(*args, **kwargs):
                """自定义的 completion wrapper，使用格式整齐的单文件 logging"""
                # 调用原始的 completion
                response = original_completion(*args, **kwargs)

                # 自定义 logging 逻辑
                try:
                    # 使用固定的文件名 completions.json
                    log_file = os.path.join(
                        self.config.log_completions_folder,
                        'completions.json'
                    )

                    # 准备要记录的数据
                    log_entry = {
                        'timestamp': time.time(),
                        'model': self.config.model,
                        'messages': kwargs.get('messages', args[1] if len(args) > 1 else None),
                        'response': response,
                        'kwargs': {
                            k: v
                            for k, v in kwargs.items()
                            if k not in ('messages', 'client')
                        },
                    }

                    # 记录到文件
                    custom_log_completion(log_file, log_entry)

                except Exception as e:
                    # 如果 logging 失败，记录错误但不影响主流程
                    logger.warning(f'Failed to log LLM completion: {e}')

                return response

            # 替换 _completion
            self._completion = custom_logging_wrapper

    # 应用 monkey patch
    LLM.__init__ = patched_init
    logger.info('Applied custom LLM completion logging (single file, formatted JSON)')
