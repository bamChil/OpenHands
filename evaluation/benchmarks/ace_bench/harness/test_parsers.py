import re


def parse_log_pytest(log: str) -> dict:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: {
            'total': int,
            'success': int,
            'failure': int,
            'pass_rate': float,
            'tests': {
                'test_name': 'PASSED' | 'FAILED' | 'SKIPPED' | 'XFAIL' | 'XPASS' | 'ERROR' | 'WARNING'
            }
        }
    """
    result = {
        'total': 0,
        'success': 0,
        'failure': 0,
        'pass_rate': 0.0,
        'tests': {}
    }

    if not log or not log.strip():
        return result

    lines = log.split('\n')

    # Find the "short test summary info" section
    summary_start_idx = -1
    for i, line in enumerate(lines):
        if 'short test summary info' in line.lower():
            summary_start_idx = i + 1  # Start from the next line
            break

    # If we didn't find the summary section, return empty result
    if summary_start_idx == -1:
        return result

    # Parse lines from the summary section until we hit a separator line or end
    for i in range(summary_start_idx, len(lines)):
        line = lines[i].strip()

        # Stop if we hit a separator line (starts with ===)
        if line.startswith('==='):
            break

        # Skip empty lines
        if not line:
            continue

        # Match PASSED
        if line.startswith('PASSED '):
            match = re.match(r'PASSED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'PASSED'
                result['success'] += 1
                result['total'] += 1

        # Match FAILED
        elif line.startswith('FAILED '):
            match = re.match(r'FAILED\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'FAILED'
                result['failure'] += 1
                result['total'] += 1

        # Match SKIPPED
        elif line.startswith('SKIPPED '):
            # Format: SKIPPED [1] test_example.py:15: reason
            match = re.match(r'SKIPPED\s+\[\d+\]\s+(.+?)(?::\d+)?:\s+.*$', line)
            if match:
                test_path = match.group(1).strip()
                # Convert path format to test name format if needed
                # For now, use the path as is
                result['tests'][test_path] = 'SKIPPED'
                result['success'] += 1
                result['total'] += 1

        # Match XFAIL (expected failure - counts as success)
        elif line.startswith('XFAIL '):
            match = re.match(r'XFAIL\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'XFAIL'
                result['success'] += 1
                result['total'] += 1

        # Match XPASS (unexpected pass - counts as failure)
        elif line.startswith('XPASS '):
            match = re.match(r'XPASS\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'XPASS'
                result['failure'] += 1
                result['total'] += 1

        # Match ERROR
        elif line.startswith('ERROR '):
            match = re.match(r'ERROR\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'ERROR'
                result['failure'] += 1
                result['total'] += 1

        # Match WARNING (counts as success)
        elif line.startswith('WARNING '):
            match = re.match(r'WARNING\s+(.+?)(?:\s+-\s+.*)?$', line)
            if match:
                test_name = match.group(1).strip()
                result['tests'][test_name] = 'WARNING'
                result['success'] += 1
                result['total'] += 1

    # Calculate pass rate
    if result['total'] > 0:
        result['pass_rate'] = round(result['success'] / result['total'], 4)

    return result



MAP_REPO_TO_PARSER = {
    "linkedin/Liger-Kernel": parse_log_pytest,
}

MAP_REPO_TO_TEST_CMD = {
    "linkedin/Liger-Kernel": "pytest -rA -p no:cacheprovider",
}
