import os
import json
import peft
import torch
from transformers import AutoModelForCausalLM


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path

    def read_last(self):
        if not os.path.isfile(self.log_path):
            return {}
        with open(self.log_path, "r") as jsl:
            js_str = jsl.readlines()[-1]
        js_dict = json.loads(js_str)
        return js_dict

    def append(self, data):
        js_str = json.dumps(data)
        with open(self.log_path, "a") as jsl:
            jsl.write(js_str + "\n")


def init_lora(modelPath, loraPath, peftConfig=None):
    if peftConfig is None:
        # 默认的config就是r=8/16, alpha=r/2r
        peftConfig = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    else:
        peftConfig = peft.LoraConfig(**peftConfig)
    model = AutoModelForCausalLM.from_pretrained(modelPath).to("cpu")

    os.makedirs(loraPath, exist_ok=True)
    model.add_adapter(peftConfig, adapter_name="init_lora")
    model.save_pretrained(loraPath)


def get_per_token_logps(logits, input_ids):
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    per_token_logps = []  # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(
            log_probs, dim=1, index=input_ids_row.unsqueeze(1)
        ).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def grpo_loss(
    per_token_logps,
    ref_logprobs,
    rollout_logprobs,
    training_mask,
    reward,
    epsilon,
    beta,
    G,
):
    pi_theta = torch.exp(per_token_logps)
    pi_old = torch.exp(rollout_logprobs)
    pi_ref = torch.exp(ref_logprobs)
    f = pi_theta / pi_old
    restricted_f = torch.min(
        f * reward, torch.clamp(f, 1 - epsilon, 1 + epsilon) * reward
    )
    D_kl = pi_ref / pi_theta - ref_logprobs + per_token_logps - 1
    sum_val = torch.sum(restricted_f + beta * D_kl * training_mask, -1)
    avg_val = sum_val / torch.sum(training_mask, -1)
    return -torch.sum(avg_val) / G
