import ast
import sys
from io import StringIO


def execute_llm_code(code_string):
    # Parse the code string to check for syntax errors
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        return f"Syntax error in the generated code: {str(e)}"

    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        # Execute the code
        exec(code_string)
        output = redirected_output.getvalue()
        return output
    except Exception as e:
        return f"Error executing the code: {str(e)}"
    finally:
        # Restore stdout
        sys.stdout = old_stdout
