import os
import ray
from ray.util.queue import Queue
import torch


def data_process(tokenizer, data, max_model_len, device):
    batch = []
    mask = []
    ref_logprobs = []
    rollout_logprobs = []
    reward = []
    for d in data:
        batch.append([item["token_id"] for item in d["token_info"]])
        mask.append([item["mask"] for item in d["token_info"]])
        ref_logprobs.append(
            [item["ref_logprob"] for item in d["token_info"][1:]]
            + [0 for _ in range(max_model_len - len(d["token_info"]))]
        )
        rollout_logprobs.append(
            [item["rollout_logprob"] for item in d["token_info"][1:]]
            + [0 for _ in range(max_model_len - len(d["token_info"]))]
        )
        reward.append([d["reward"]])

    # 这个获得的是attention mask
    batch_sentences = {
        "input_ids": batch,
    }
    padded = tokenizer.pad(
        batch_sentences,
        padding="max_length",
        max_length=max_model_len,
    )
    batch = padded["input_ids"]
    attn_mask = padded["attention_mask"]

    # 这个获得的是训练的mask, 用于交叉熵损失
    batch_sentences["attention_mask"] = mask
    padded = tokenizer.pad(
        batch_sentences,
        padding="max_length",
        max_length=max_model_len,
    )
    training_mask = padded["attention_mask"]

    batch = torch.tensor(batch, dtype=torch.int64).to(device)
    attn_mask = torch.tensor(attn_mask, dtype=torch.bool).to(device)
    training_mask = torch.tensor(training_mask, dtype=torch.bool).to(device)
    ref_logprobs = torch.tensor(ref_logprobs, dtype=torch.bfloat16).to(device)
    rollout_logprobs = torch.tensor(rollout_logprobs, dtype=torch.bfloat16).to(device)
    reward = torch.tensor(reward, dtype=torch.bfloat16).to(device)

    # FIXME: 这里的ref_logprobs和rollout_logprobs没有补全到和那啥一样的长度
    return (
        batch,
        attn_mask,
        training_mask[:, 1:],
        ref_logprobs,
        rollout_logprobs,
        reward,
    )


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


# def trainer(
#     manager,
#     model_path,
#     lora_path,
#     ds_config,
#     devices,
#     total_batch,
#     out_queue,
#     max_model_len,
#     epsilon,
#     beta,
#     G,
# ):
#     out_queue = Queue()
#     import deepspeed
#     import torch.distributed as dist
#
#     deepspeed.init_distributed(dist_backend="nccl")
#
#     from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
#
#     nf4_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path, quantization_config=nf4_config
#     )
#     model.load_adapter(lora_path, is_trainable=True)
#     engine, optimizer, _, _ = deepspeed.initialize(
#         model=model,
#         config=ds_config,
#         model_parameters=model.parameters(),
#         dist_init_required=False,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     world = dist.get_world_size()
#
#     while True:
#         batch_count = 0
#         while True:
#             curr_batch = out_queue.get()
#             batch_count += len(curr_batch) * world
#             data, attn_mask, training_mask, ref_logprobs, rollout_logprobs, reward = (
#                 data_process(tokenizer, curr_batch, max_model_len, engine.device)
#             )
#             outputs = engine(
#                 data,
#                 attn_mask,
#             )
#             per_token_logps = get_per_token_logps(outputs.logits, data)
#             loss = grpo_loss(
#                 per_token_logps,
#                 ref_logprobs,
#                 rollout_logprobs,
#                 training_mask,
#                 reward,
#                 epsilon,
#                 beta,
#                 G,
#             )
#             engine.backward(loss)
#             engine.step()
#
#             if batch_count >= total_batch:
#                 break
#         engine.model.save_pretrained(lora_path)


@ray.remote
class TrainerWorker:
    def __init__(self, model_path, lora_path, ds_config, devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "11451"
        import deepspeed

        deepspeed.init_distributed(dist_backend="nccl")
        import torch.distributed as dist
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.lora_path = lora_path
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=nf4_config
        )
        model.load_adapter(lora_path, is_trainable=True)
        self.engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.world = dist.get_world_size()
        print(self.world)

    def data_process(self, data, max_model_len, device):
        batch = []
        mask = []
        ref_logprobs = []
        rollout_logprobs = []
        reward = []
        for d in data:
            batch.append([item["token_id"] for item in d["token_info"]])
            mask.append([item["mask"] for item in d["token_info"]])
            ref_logprobs.append(
                [item["ref_logprob"] for item in d["token_info"][1:]]
                + [0 for _ in range(max_model_len - len(d["token_info"]))]
            )
            rollout_logprobs.append(
                [item["rollout_logprob"] for item in d["token_info"][1:]]
                + [0 for _ in range(max_model_len - len(d["token_info"]))]
            )
            reward.append([d["reward"]])

        # 这个获得的是attention mask
        batch_sentences = {
            "input_ids": batch,
        }
        padded = self.tokenizer.pad(
            batch_sentences,
            padding="max_length",
            max_length=max_model_len,
        )
        batch = padded["input_ids"]
        attn_mask = padded["attention_mask"]

        # 这个获得的是训练的mask, 用于交叉熵损失
        batch_sentences["attention_mask"] = mask
        padded = self.tokenizer.pad(
            batch_sentences,
            padding="max_length",
            max_length=max_model_len,
        )
        training_mask = padded["attention_mask"]

        batch = torch.tensor(batch, dtype=torch.int64).to(device)
        attn_mask = torch.tensor(attn_mask, dtype=torch.bool).to(device)
        training_mask = torch.tensor(training_mask, dtype=torch.bool).to(device)
        ref_logprobs = torch.tensor(ref_logprobs, dtype=torch.bfloat16).to(device)
        rollout_logprobs = torch.tensor(rollout_logprobs, dtype=torch.bfloat16).to(
            device
        )
        reward = torch.tensor(reward, dtype=torch.bfloat16).to(device)

        # FIXME: 这里的ref_logprobs和rollout_logprobs没有补全到和那啥一样的长度
        return (
            batch,
            attn_mask,
            training_mask[:, 1:],
            ref_logprobs,
            rollout_logprobs,
            reward,
        )

    def step(self, total_batch, out_queue, max_model_len, epsilon, beta, G):
        batch_count = 0
        while True:
            curr_batch = out_queue.get()
            batch_count += len(curr_batch) * self.world
            data, attn_mask, training_mask, ref_logprobs, rollout_logprobs, reward = (
                self.data_process(curr_batch, max_model_len, self.engine.device)
            )
            outputs = self.engine(
                data,
                attn_mask,
            )
            per_token_logps = get_per_token_logps(outputs.logits, data)
            loss = grpo_loss(
                per_token_logps,
                ref_logprobs,
                rollout_logprobs,
                training_mask,
                reward,
                epsilon,
                beta,
                G,
            )
            self.engine.backward(loss)
            self.engine.step()

            if batch_count >= total_batch:
                break

    def save_lora(self):
        self.engine.model.save_pretrained(self.lora_path)
        return True
