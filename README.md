# LiteChainRL

A lightweight, multi-turn RL framework leveraging LoRA/QLoRA for efficient agent training.

## Overview

LiteChainRL streamlines experimentation with reinforcement learning on modern LLMs by minimizing GPU requirements and dependencies. It provides:

- **Multi‑turn Rollouts** for interactive tasks such as function calling and code generation

- **QLoRA Integration** to dramatically reduce memory usage through low‑rank adapter tuning

- **Easy Setup** with clear abstractions and a plug‑and‑play “Trigger” interface for custom environments

With LiteChainRL, researchers can rapidly prototype agent behaviors without the overhead of full‑parameter distributed training.

## Structure

LiteChainRL aims to simplify RL agent prototype build. Therefore, we keep our workflow simple and easy to adapt. It only contains a Rollout Manager and Trainer Manager. The rollout sample send to a queue then accessed by the Trainer Manager. Once Rollout Manager needs to update its LoRA adapter weight. It will load the weight from the disk.

![LiteChainRL Structure](image/structure.svg)

## Key Features

1. **Multi-Turn Rollouts**

Initiate sequential interactions by defining your own "Trigger" class as an environment. Whenever a specified pattern appears in the model’s output, LiteChainRL pauses generation, invokes the environment logic, and resumes—enabling complex, agent‑style workflows.

2. **LoRA/QLoRA Support**

Leverage low‑rank adapters to fine‑tune large models with a fraction of the memory footprint required for full‑parameter updates.

3. **Lightweight Design**

- Rollouts & Log-Probs: Powered by vLLM for blazing‑fast sample generation
- Training Backend: Using Deepspeed to optimize GPU memory usage and Ray to build a tiny training cluster
- Rapid Prototyping: High‑level abstractions let you focus on ideas, not infrastructure

4. **Agent-Ready Abstraction**

The extendable Trigger system lets you connect any callable environment—be it function execution, API calls, or custom simulator—to the model’s generation loop, making LiteChainRL an ideal playground for building interactive agents.

## Usage

**The LiteChainRL is still under development and may have some unknown bugs. Any issues and pull requests are welcome.** Currently it only supports GRPO and vLLM-0.7.3. The higher version vLLM can still run the `LoRAEngine` in the `rollout/lora_engine.py` but have generation bugs.

To prototype your agent with our project. You just need to do 5 steps.

1. Define your own `Trigger`.
2. Define your own `Dataset`.
3. Define your own `Reward`.
4. Import your own `Dataset` and `Reward` in the `train.py`.
5. Set the configuration file and launch the training.

### Trigger

Define your own `Trigger` based on `BaseTrigger` class in `trigger/base_trigger.py`. Your `Trigger` should contain `trigger` and `copy` method. The `trigger` method will get every step generation of the rollout LLM as parameter. If the generation fulfill your patterns, the `trigger` method should return a string that append to the generation. The `LoRAEngine` will stop the generation request, add your new string after it, and send new generation request to the vLLM engine. As for the `copy` method, it should return a new `Trigger` instance. Since during the sampling step in the rollout, it is necessary for every prompt to hold its own `Trigger`.

You can also refer to `trigger/code_trigger.py`. The `CodeTrigger` in it will stop the generation when Python code block detected. Then it will execute it, catch the printed output and wrap the output with `<Code Result></Code Result>`. Similary, you can design something like `FunctionCallTrigger` that can detect function call pattern in the LLM generation and send request to a remote server and append the response to LLM generation.

There are also a `NoneTrigger` in `trigger/none_trigger.py`. It always return None. So it will never append new text to a generation. It is used for single-turn generation.

### Dataset

Your should define your own `Dataset` based on `torch.utils.data.Dataset`. The `__getitem__` method of your own `Dataset` should return a triplet. The frist is a prompt. The second is a `Trigger` instance that uses your designed `Trigger` in the earlier step. The third is the `metadata`, which will be used for `Reward` later.

### Reward

Currently `Reward` only supports rule-based reward and `LoRAEngine` reward. The latter one is use the base model you choose to join the reward step. Therefore, you need to remain a argument in the `__init__` method for pass the `LoRAEngine` instance. Then you need to define the `reward(self, items, metadata)` method.

The example shows in  `reward/code_eval_reward.py`. You can check it for more implementation details.

### Train

Edit the `train.py` to import your own `Dataset` and `Reward`. Then instanciate your `Dataset`. The detailed location marks with `# NOTE: ` in the file.

Edit `ds_config.json` for Deepspeed config, `lora_config.json` for LoRA config and `train.sh` for training config. Then lauch with

```bash
bash train.sh
```

## Example--Reinplementation of TinyZero with LiteChainRL

The TinyZero is a simple RL project. We reimplemented it here as an simple example. The detail of TinyZero task can be founded here.

[TinyZero](https://github.com/Jiayi-Pan/TinyZero)

The task is simple. However, there is a high risk for LLM to calculate a mathematical expression incorrectly. Therefore, the best idea is to give it a "calculator". In other worlds, the LLM need to use the "calculator" as a tool to implement complex tasks.

Training on 1024 samples takes about 13 hours with 2 H20 GPUs. One H20 for vLLM rollout generation and one H20 for training. More GPUs will accelerate the training process.
