import os
import vllm
from uuid import uuid4
from dataclasses import dataclass
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest


@dataclass
class LoRAEngineConfig:
    model: str
    lora_path: str
    cuda_visible_devices: str = "0"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    max_token_per_turn: int = 1024
    seed: int = 42
    qlora: bool = False


class LoRAEngine:
    def __init__(self, config: LoRAEngineConfig):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        self.max_model_len = config.max_model_len
        if config.qlora:
            engine_args = vllm.EngineArgs(
                model=config.model,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_prefix_caching=True,
                max_model_len=config.max_model_len,
                tensor_parallel_size=len(config.cuda_visible_devices.split(",")),
                enable_lora=True,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            )
        else:
            engine_args = vllm.EngineArgs(
                model=config.model,
                gpu_memory_utilization=config.gpu_memory_utilization,
                enable_prefix_caching=True,
                max_model_len=config.max_model_len,
                tensor_parallel_size=len(config.cuda_visible_devices.split(",")),
                enable_lora=True,
            )
        if vllm.__version__.startswith("0.8"):
            from vllm.v1.engine.llm_engine import LLMEngine
        else:
            from vllm.engine.llm_engine import LLMEngine
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)

        # FIXME: Currently seed is not compatible with the multi-turn generation.
        self.params = vllm.SamplingParams(
            temperature=1,
            repetition_penalty=1,
            max_tokens=config.max_token_per_turn,
            # seed=config.seed,
        )
        self.logprob_params = vllm.SamplingParams(
            temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1
        )
        self.normal_params = vllm.SamplingParams(
            temperature=0.8,
            repetition_penalty=1,
            max_tokens=config.max_token_per_turn,
            # seed=config.seed,
        )

        self.lora_idx = 1
        self.lora_request = LoRARequest(
            f"lora_{self.lora_idx}", self.lora_idx, config.lora_path
        )

    def update_lora(self, lora_path):
        self.engine.remove_lora(self.lora_idx)
        self.lora_idx += 1
        self.lora_request = LoRARequest(
            f"lora_{self.lora_idx}", self.lora_idx, lora_path
        )

    def single_turn_gen(self, prompts):
        results = {}
        for prompt in prompts:
            request_id = str(uuid4())
            results[request_id] = None
            self.engine.add_request(request_id, prompt, self.normal_params)
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for op in outputs:
                if op.finished:
                    request_id = op.request_id
                    results[request_id] = op.prompt + op.outputs[0].text
        return list(results.values())

    def single_logprobs(self, tokens, lora=True):
        k = str(uuid4())
        self.engine.reset_prefix_cache()
        token_ids = [it["token_id"] for it in tokens]
        prompt = vllm.TokensPrompt(prompt_token_ids=token_ids)
        if lora:
            self.engine.add_request(
                k,
                prompt,
                self.logprob_params,
                lora_request=self.lora_request,
            )
        else:
            self.engine.add_request(
                k,
                prompt,
                self.logprob_params,
            )

        data = [None]  # 第一个token没有logprobs
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for op in outputs:
                if op.finished:
                    request_id = op.request_id
                    prompt_token_ids = list(op.prompt_token_ids)
                    lps = op.prompt_logprobs

                    for pid, lp in zip(prompt_token_ids[1:], lps[1:]):
                        vals = list(lp.values())[0]
                        data.append(
                            {
                                "token_id": pid,
                                "logprob": vals.logprob,
                                "rank": vals.rank,
                            }
                        )
        return data

    def logprobs(self, data, ref=True):
        for _, v in data.items():
            rollout_logprobs = self.single_logprobs(v, True)
            for i in range(1, len(v)):
                v[i]["rollout_token_id"] = rollout_logprobs[i]["token_id"]
                v[i]["rollout_logprob"] = rollout_logprobs[i]["logprob"]
                v[i]["rollout_rank"] = rollout_logprobs[i]["rank"]
            if ref:
                ref_logprobs = self.single_logprobs(v, False)
                for i in range(1, len(v)):
                    v[i]["ref_token_id"] = ref_logprobs[i]["token_id"]
                    v[i]["ref_logprob"] = ref_logprobs[i]["logprob"]
                    v[i]["ref_rank"] = ref_logprobs[i]["rank"]
        return data

    def multi_turn_gen(self, prompts, envs, ref=True):
        """
        For given input and trigger environments, do the generation and calculate every token's logprob
        and training mask. Only model outputs will be masked as 1. Input prompt and trigger generated
        text will be masked as 0.

        :prompt:
        """
        results = self._multi_turn_gen(prompts, envs)
        data = self.logprobs(results, ref)
        for k, v in data.items():
            data[k] = {
                "text": self.tokenizer.decode([it["token_id"] for it in v]),
                "token_info": v,
            }
        return list(data.values())

    def _multi_turn_gen(self, prompts, envs):
        envMap = {}
        tokenInfo = {}
        for prompt, env in zip(prompts, envs):
            request_id = str(uuid4())
            envMap[request_id] = env
            tokenInfo[request_id] = []
            self.engine.add_request(
                f"0-{request_id}",
                prompt,
                self.params,
                lora_request=self.lora_request,
            )

        finishedRequest = {}
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for op in outputs:
                prompt = op.prompt + op.outputs[0].text
                full_request_id = op.request_id
                request_count, request_id = full_request_id.split("-", 1)
                request_count = int(request_count)
                prompt_token_ids = list(op.prompt_token_ids)
                output_token_ids = list(op.outputs[0].token_ids)

                env_output = envMap[request_id].trigger(prompt)
                if env_output:
                    # 这个传参只能是列表, 单元素会导致abort不掉
                    self.engine.abort_request([full_request_id])
                    tokenInfo[request_id].append(
                        (
                            len(prompt_token_ids),
                            len(output_token_ids),
                        )
                    )
                    # 有极小的概率生成+环境生成大于max model len
                    if (
                        len(self.tokenizer.encode(prompt + env_output))
                        >= self.max_model_len
                    ):
                        finishedRequest[request_id] = (
                            prompt_token_ids + output_token_ids
                        )
                        continue
                    self.engine.add_request(
                        f"{request_count + 1}-{request_id}",
                        prompt + env_output,
                        self.params,
                        lora_request=self.lora_request,
                    )
                if op.finished:
                    finishedRequest[request_id] = prompt_token_ids + output_token_ids
                    tokenInfo[request_id].append(
                        (
                            len(prompt_token_ids),
                            len(output_token_ids),
                        )
                    )
                if len(prompt_token_ids) + len(output_token_ids) > self.max_model_len:
                    finishedRequest[request_id] = prompt_token_ids + output_token_ids
                    tokenInfo[request_id].append(
                        (
                            len(prompt_token_ids),
                            len(output_token_ids),
                        )
                    )
                    self.engine.abort_request(full_request_id)

        masks = {}
        for k, v in tokenInfo.items():
            masks[k] = []
            masks[k] += [0] * v[0][0] + [1] * v[0][1]
            for sz0, sz1 in v[1:]:
                masks[k] += [0] * (sz0 - len(masks[k])) + [1] * sz1

        result = {}
        for k in tokenInfo.keys():
            result[k] = []
            for t, m in zip(finishedRequest[k], masks[k]):
                result[k].append(
                    {
                        "token": self.tokenizer.convert_ids_to_tokens(t),
                        "token_id": t,
                        "mask": m,
                    }
                )
            # 这里少一个预留给生成logprobs
            result[k] = result[k][: self.max_model_len - 1]
        return result
