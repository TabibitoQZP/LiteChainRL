import re
import numpy as np
from reward.base_reward import BaseReward


class TinyZeroReward(BaseReward):
    def __init__(self, engine_instance):
        pass

    @staticmethod
    def extract_answer(text):
        pat = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if len(pat) <= 3:
            return None
        else:
            return pat[-1]

    @staticmethod
    def verify(ans, metadata):
        try:
            result = eval(ans)
            if result != metadata[1]:
                return 0
            for m in metadata[0]:
                if str(m) not in ans:
                    return 0
            return 1
        except:
            return 0

    def reward(self, items, metadata):
        texts = [item["text"] for item in items]
        sz = len(texts)

        rewards = []
        for i in range(sz):
            text = texts[i]
            ans = TinyZeroReward.extract_answer(text)
            print(ans)

            if ans is None:
                rewards.append(0)
                continue

            rewards.append(1 + TinyZeroReward.verify(ans, metadata[i]))

        std = np.std(rewards)
        avg = np.mean(rewards)
        for i in range(len(items)):
            if std == 0:
                items[i]["reward"] = 0
            else:
                items[i]["reward"] = (rewards[i] - avg) / std
        return items
