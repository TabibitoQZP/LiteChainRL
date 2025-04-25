import re
import io
import sys
import signal

from trigger.base_trigger import BaseTrigger


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out!")


def removeIndent(text: str):
    lines = text.splitlines()
    if len(lines) == 0:
        return text
    cnt = 0
    while lines[0][cnt:].startswith(" ") or lines[0][cnt:].startswith("\t"):
        cnt += 1
    for i in range(len(lines)):
        lines[i] = lines[i][cnt:]
    return "\n".join(lines)


def extractFinalCode(text: str):
    pattern = r"```python\n(.*?)```"
    code_blocks = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if code_blocks:
        return removeIndent(code_blocks[-1])
    return ""


def execWithOutput(code, env, timeLimit=8):
    cap = io.StringIO()
    sys.stdout = cap

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeLimit)
    try:
        exec(code, env)
    except TimeoutError as e:
        sys.stdout = sys.__stdout__
        signal.alarm(0)
        return f"Timeout: Runing over {timeLimit} seconds and not stop. Please check your code!"
    except Exception as e:
        sys.stdout = sys.__stdout__
        signal.alarm(0)
        return f"Error: {str(e)}\n"
    sys.stdout = sys.__stdout__
    signal.alarm(0)
    return cap.getvalue()


class CodeTrigger(BaseTrigger):
    def __init__(self, init_env: dict, max_try=16, timeLimit=8):
        self.env = init_env
        self.max_try = max_try
        self.timeLimit = timeLimit

    def trigger(self, input_text: str):
        if not input_text.strip().endswith("```") or input_text.count("```") % 2 != 0:
            return None
        code = extractFinalCode(input_text)
        outputs = ""
        if input_text.count("```") // 2 > self.max_try:
            outputs = "MaxTry: You already write many code blocks, please stop and give your final result."
        outputs = execWithOutput(code, self.env, self.timeLimit)
        if outputs == "":
            outputs = "EmptyOutput: You do not print any result. Please use `print` function to output variables you want."
        return f"<Code Result>\n{outputs}</Code Result>\n"

    def copy(self):
        return CodeTrigger(self.env.copy(), self.max_try)
