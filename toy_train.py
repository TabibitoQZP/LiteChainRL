import random
from transformers import AutoTokenizer
from torch.utils.data import Dataset

prompt_template = """
You will be given a list of integers and a result integer. You need to use every element once and only once with the oprand `+` and `-` to generate a equation that equals the result integer. The following is an example:

**Integer List**: [19, 36, 55, 7]

**Result Integer**: 65

<Answer>55 + 36 - 7 -19</Answer>

You can use python code to help you find the combination. Specifically, you can write your python code wrapped in ```python```. You need to use **print** function to print the result you want. When you figure out the combination. You should write the result wrapped in <Answer></Answer>. You only need to list the combination and do not contain equal and result integer in it. The following is your task.

**Integer List**: {integer_list}

**Result Integer**: {result_integer}

Now start your analysis step by step. Try to use python code and give your final result wrapped in <Answer></Answer>.
"""


class ToyDataset(Dataset):
    def __init__(
        self,
        model,
        minval,
        maxval,
        mincount,
        maxcount,
        dataset_size,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.minval = minval
        self.maxval = maxval
        self.mincount = mincount
        self.maxcount = maxcount
        self.dataset_size = dataset_size

        self.oprands = "+ -".split()

        tokenizer = AutoTokenizer.from_pretrained(model)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt_template.strip(),
            },
        ]
        self.prompt_template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def single_gen(self):
        count = random.randint(self.mincount, self.maxcount)
        values = [random.randint(self.minval, self.maxval) for _ in range(count)]

    def _init_dataset(self):
        pass


if __name__ == "__main__":
    pass
