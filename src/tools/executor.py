"""
Python code executor for agent tool use.
Safely executes Python code and captures output.
"""

import sys
import io
import signal
import traceback
from typing import Optional
from contextlib import contextmanager


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Handler for timeout signal."""
    raise TimeoutException("Code execution timed out")


@contextmanager
def time_limit(seconds: int):
    """
    Context manager for timing out code execution.

    Args:
        seconds: Maximum execution time in seconds
    """
    # Set up signal handler for timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable alarm
        signal.alarm(0)


def execute_code(code_str: str, timeout: int = 5, max_output_length: int = 2000) -> str:
    """
    Execute Python code and capture output.

    Args:
        code_str: Python code to execute
        timeout: Maximum execution time in seconds (default: 5)
        max_output_length: Maximum length of output to return (default: 2000 chars)

    Returns:
        Output string (stdout or error traceback)

    Examples:
        >>> execute_code("print(2 + 2)")
        '4\\n'

        >>> execute_code("print(345 * 982)")
        '338790\\n'

        >>> execute_code("x = 1 / 0")
        'ZeroDivisionError: division by zero\\n'
    """
    # Create string buffer to capture stdout
    output_buffer = io.StringIO()
    old_stdout = sys.stdout

    try:
        # Redirect stdout to our buffer
        sys.stdout = output_buffer

        # Create isolated namespace for execution
        namespace = {
            '__builtins__': __builtins__,
            # Add safe imports if needed
            'math': __import__('math'),
            'random': __import__('random'),
        }

        # Execute code with timeout
        try:
            with time_limit(timeout):
                exec(code_str, namespace)
        except TimeoutException:
            return f"ERROR: Execution timed out after {timeout} seconds\n"
        except Exception as e:
            # Capture traceback for errors
            error_type = type(e).__name__
            error_msg = str(e)
            tb = traceback.format_exc()

            # Return concise error message
            if error_msg:
                return f"{error_type}: {error_msg}\n"
            else:
                return f"{error_type}\n"

        # Get captured output
        output = output_buffer.getvalue()

        # Truncate if too long
        if len(output) > max_output_length:
            output = output[:max_output_length] + "\n... (output truncated)"

        return output if output else ""

    finally:
        # Restore original stdout
        sys.stdout = old_stdout
        output_buffer.close()


def execute_code_safe(code_str: str, timeout: int = 5) -> dict:
    """
    Execute code and return structured result.

    Args:
        code_str: Python code to execute
        timeout: Maximum execution time in seconds

    Returns:
        Dictionary with keys:
            - success: bool indicating if execution succeeded
            - output: str containing output or error message
            - error: Optional error type if failed

    Examples:
        >>> result = execute_code_safe("print(42)")
        >>> result['success']
        True
        >>> result['output']
        '42\\n'
    """
    try:
        output = execute_code(code_str, timeout=timeout)

        # Check if output indicates an error
        is_error = output.startswith("ERROR:") or "Error:" in output or "Exception:" in output

        return {
            'success': not is_error,
            'output': output,
            'error': 'ExecutionError' if is_error else None
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'error': str(e)
        }


def extract_code_from_exec_tags(text: str, start_tag: str = "<EXEC>", end_tag: str = "</EXEC>") -> Optional[str]:
    """
    Extract code from between execution tags.

    Args:
        text: Text containing execution tags
        start_tag: Opening tag (default: "<EXEC>")
        end_tag: Closing tag (default: "</EXEC>")

    Returns:
        Extracted code string or None if tags not found

    Examples:
        >>> text = "Let me calculate: <EXEC>print(2+2)</EXEC>"
        >>> extract_code_from_exec_tags(text)
        'print(2+2)'
    """
    start_idx = text.find(start_tag)
    end_idx = text.find(end_tag)

    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None

    # Extract code between tags
    code = text[start_idx + len(start_tag):end_idx]
    return code.strip()


if __name__ == "__main__":
    # Test the executor
    print("=== Testing Python Code Executor ===\n")

    # Test 1: Simple arithmetic
    print("Test 1: Simple arithmetic")
    code1 = "print(345 * 982)"
    result1 = execute_code(code1)
    print(f"Code: {code1}")
    print(f"Output: {result1}")

    # Test 2: Multiple operations
    print("\nTest 2: Multiple operations")
    code2 = """
x = 10
y = 20
print(f"Sum: {x + y}")
print(f"Product: {x * y}")
"""
    result2 = execute_code(code2)
    print(f"Output:\n{result2}")

    # Test 3: Error handling
    print("\nTest 3: Error handling")
    code3 = "x = 1 / 0"
    result3 = execute_code(code3)
    print(f"Code: {code3}")
    print(f"Output: {result3}")

    # Test 4: Extract code from tags
    print("\nTest 4: Extract code from tags")
    text = "Let me calculate: <EXEC>print(2**10)</EXEC> Done!"
    extracted = extract_code_from_exec_tags(text)
    print(f"Text: {text}")
    print(f"Extracted: {extracted}")
    if extracted:
        result = execute_code(extracted)
        print(f"Result: {result}")
