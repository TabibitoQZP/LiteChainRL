import os
import vllm
from uuid import uuid4
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest


class LoRAEngine:
    def __init__(
        self,
        model,
        cuda_visible_devices,
        lora_path,
        max_model_len=4096,
        gpu_memory_utilization=0.7,
        max_token_per_turn=1024,
        seed=42,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.max_model_len = max_model_len
        engine_args = vllm.EngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            max_model_len=max_model_len,
            tensor_parallel_size=len(cuda_visible_devices.split(",")),
            enable_lora=True,
        )
        self.engine = vllm.LLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.params = vllm.SamplingParams(
            temperature=0.8,
            repetition_penalty=1,
            max_tokens=max_token_per_turn,
            seed=seed,
        )
        self.logprob_params = vllm.SamplingParams(
            temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1
        )
        self.normal_params = vllm.SamplingParams(
            temperature=0.8,
            repetition_penalty=1,
            max_tokens=max_token_per_turn,
            seed=seed,
        )

        self.lora_idx = 1
        self.lora_request = LoRARequest(
            f"lora_{self.lora_idx}", self.lora_idx, lora_path
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
        return data

    def _multi_turn_gen(self, prompts, envs):
        envMap = {}
        tokenInfo = {}
        for prompt, env in zip(prompts, envs):
            request_id = str(uuid4())
            envMap[request_id] = env
            tokenInfo[request_id] = []
            self.engine.add_request(
                request_id, prompt, self.params, lora_request=self.lora_request
            )

        finishedRequest = {}
        while self.engine.has_unfinished_requests():
            outputs = self.engine.step()
            for op in outputs:
                prompt = op.prompt + op.outputs[0].text
                request_id = op.request_id
                prompt_token_ids = list(op.prompt_token_ids)
                output_token_ids = list(op.outputs[0].token_ids)

                env_output = envMap[request_id].trigger(prompt)
                if env_output:
                    self.engine.abort_request(request_id)
                    self.engine.add_request(
                        request_id,
                        prompt + env_output,
                        self.params,
                        lora_request=self.lora_request,
                    )
                    tokenInfo[request_id].append(
                        (
                            len(prompt_token_ids),
                            len(output_token_ids),
                        )
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
                    self.engine.abort_request(request_id)

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
