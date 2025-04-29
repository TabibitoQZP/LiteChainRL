import random
from torch.utils.data import Dataset
from trigger.none_trigger import NoneTrigger
from trigger.tinyzero_tirgger import TinyZeroTrigger

prompt_template = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Using the numbers {numbers}, `+` and `-` to create an equation that equals {result}. Every number in the list should be used once. You need to wrap your final result in the `<answer></answer>` tag. For example

<answer>55 + 36 - 7 -19</answer>

Now please find your final answer step by step.<|im_end|>
<|im_start|>assistant
"""

agent_version = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Using the numbers {numbers}, `+` and `-` to create an equation that equals {result}. Every number in the list should be used once. You need to wrap your final result in the `<answer></answer>` tag. For example

<answer>55 + 36 - 7 -19</answer>

You can use tool to help you do the calculation. Specifically, once you wrap a mathematical expression in `<expression></expression>`, then it will be automatically implemented and return the result wrapped in `<result></result>`. The following is an example.

<expression>1 + 1</expression>
<result>2</result>

Now please use these feature to help you find your final answer step by step. Do not forget to wrap your final expression in `<answer></answer>`.<|im_end|>
<|im_start|>assistant
"""

# 不需要think则添加上这一点
# prompt_template += "<think>\n\n</think>"


class TinyZeroDataset(Dataset):
    def __init__(
        self,
        data_range,
        item_range,
        data_szie,
        seed,
        agent=False,
    ):
        self.agent = agent
        self.data_size = data_szie
        self.data = TinyZeroDataset.init_data(
            data_range,
            item_range,
            data_szie,
            seed,
        )

    @staticmethod
    def single_item_init(data_range, item_range):
        item_size = random.randint(*item_range)
        item = [random.randint(*data_range) for _ in range(item_size)]
        item_str = str(item[0])
        for i in range(1, item_size):
            item_str += random.choice(["+", "-"]) + str(item[i])
        result = eval(item_str)
        random.shuffle(item)
        return item, result

    @staticmethod
    def init_data(
        data_range,
        item_range,
        data_szie,
        seed,
    ):
        random.seed(seed)
        data = []
        for _ in range(data_szie):
            data.append(TinyZeroDataset.single_item_init(data_range, item_range))
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.agent:
            prompt = agent_version.format(numbers=str(item[0]), result=str(item[1]))
            trigger = TinyZeroTrigger()
        else:
            prompt = prompt_template.format(numbers=str(item[0]), result=str(item[1]))
            trigger = NoneTrigger()
        metadata = item
        return prompt, trigger, metadata

    def __len__(self):
        return self.data_size


if __name__ == "__main__":
    tzd = TinyZeroDataset(
        (16, 64),
        (4, 8),
        1024,
        114514,
    )
    item = tzd[0]
    print(item[0])
    print(item[1])
    print(item[2])
