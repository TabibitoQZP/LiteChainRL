import re
from trigger.base_trigger import BaseTrigger


class TinyZeroTrigger(BaseTrigger):
    def __init__(self):
        self.pattern = r"<expression>(.*?)</expression>"
        pass

    def extractFinalCode(self, text: str):
        code_blocks = re.findall(self.pattern, text, re.IGNORECASE)
        if code_blocks:
            return code_blocks[-1]
        return None

    def trigger(self, input_text: str):
        if not input_text.strip().endswith("</expression>"):
            return None
        equation = self.extractFinalCode(input_text)
        if equation is None:
            return None
        try:
            result = eval(equation)
            return f"<result>{result}</result>"
        except:
            return f"<result>Execution failed, please input a mathematical expression instead of something like equation.</result>"

    def copy(self):
        return TinyZeroTrigger()
