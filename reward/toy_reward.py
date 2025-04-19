import re
from reward.base_reward import BaseReward


class ToyReward(BaseReward):
    def __init__(self, engine_instance):
        pass

    @staticmethod
    def extract_answer(text):
        pat = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
        if len(pat) <= 1:
            return None
        else:
            return pat[-1]

    def single_reward(self, text: str, metadatum):
        answer = ToyReward.extract_answer(text)

        # without answer format, the reward is 0.
        if answer is None:
            return 0
        processed_answer = re.sub(r"\s+", "", answer)

        # the equation can not be calculate, the reward is 0.
        try:
            result = eval(processed_answer)
        except:
            return 0

        # result is not same.
        if result != metadatum["result"]:
            return 0
        values = re.split(r"[+-]", processed_answer)
        values.sort()
        gt_values = [str(item) for item in metadatum["values"]]
        gt_values.sort()

        # equation need contain all values
        if "".join(gt_values) != "".join(values):
            return 0

        return 1

    def reward(self, items, metadata):
        for item, metadatum in zip(items, metadata):
            item["reward"] = self.single_reward(item["text"], metadatum)
        return items
