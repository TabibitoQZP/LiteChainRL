import re
import numpy as np
from transformers import AutoTokenizer
from reward.base_reward import BaseReward
from rollout.lora_engine import LoRAEngine

prompt_template = """
There are many code blocks accompanied with their execution results. After these codes, there is a conclusion based on the results.

Please analyze whether the conclusion is correct by examining the code blocks and the running results.

The code blocks and results are provided below:

```python
{codeblocks}
```

**Conclusion**:
{conclusion}

# Task

- Carefully analyze the code logic, the operations performed, and the results produced.

- Compare the results with the stated conclusion to determine whether it accurately reflects the outcome of the executed code.

- Consider any possible errors, inconsistencies, or misinterpretations in the conclusion.

# Format

After completing the analysis, please provide a final verdict using the following format:

<Answer>True/False</Answer>

Now please do your analysis step by step!
"""


class CodeEvalReward(BaseReward):
    def __init__(self, engine_instance: LoRAEngine):
        self.engine_instance = engine_instance

        model = engine_instance.engine.model_config.model
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

    @staticmethod
    def extract_answer(text):
        pat = re.findall(r"<Answer>(.*?)</Answer>", text, re.DOTALL | re.IGNORECASE)
        if len(pat) <= 1:
            return None
        else:
            return pat[-1]

    def reward_prompt(self, text):
        ans = CodeEvalReward.extract_answer(text)
        cr = CodeEvalReward.extract_code_and_result(text)

        prompt = None
        if ans is not None:
            prompt = self.prompt_template.format(
                codeblocks="\n\n".join(cr), conclusion=ans
            )

        return prompt

    @staticmethod
    def resultCheck(text):
        pat = re.findall(r"<Answer>\s*(True|False)\s*</Answer>", text, re.IGNORECASE)

        if pat:
            return 0 if pat[0].lower() == "false" else 1
        return 0

    @staticmethod
    def extract_code_and_result(text):
        pat = re.findall(
            r"```python\n(.*?)```\s*<Code Result>\n(.*?)\n</Code Result>",
            text,
            re.DOTALL | re.IGNORECASE,
        )

        cr_list = []
        for p in pat:
            code = p[0]
            result = p[1]
            if result.startswith("Out of Limit") or result.startswith("Error"):
                continue
            cr_list.append(f'# Code\n{code}\n\n# Result\n"""\n{result}\n"""')
        return cr_list

    def reward(self, items, ground_truth):
        texts = [item["text"] for item in items]

        prompts = []
        for text in texts:
            prompt = self.reward_prompt(text)
            prompts.append(prompt)

        # 没有<Answer></Answer>包围的reward为0且不需要判定
        rewards = [1 for _ in prompts]
        filted_prompts = []
        for i in range(len(prompts)):
            p = prompts[i]
            if p is None:
                rewards[i] = 0
            else:
                filted_prompts.append(p)

        responses = self.engine_instance.single_turn_gen(filted_prompts)
        res_rewards = [CodeEvalReward.resultCheck(item) for item in responses]
        rr_pointer = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                # 如果结论和代码一致额外获得1分
                rewards[i] += res_rewards[rr_pointer]
                rr_pointer += 1

        for i in range(len(items)):
            items[i]["reward"] = rewards[i]
        return items
