# LiteChainRL

A lightweight, multi-turn RL framework leveraging LoRA/QLoRA for efficient LLM training.

## Overview

LiteChainRL streamlines experimentation with reinforcement learning on modern LLMs by minimizing GPU requirements and dependencies. It provides:

- **Multi‑turn Rollouts** for interactive tasks such as function calling and code generation

- **QLoRA Integration** to dramatically reduce memory usage through low‑rank adapter tuning

- **Easy Setup** with clear abstractions and a plug‑and‑play “Trigger” interface for custom environments

With LiteChainRL, researchers can rapidly prototype agent behaviors without the overhead of full‑parameter distributed training.

## Key Features

1. **Multi-Turn Rollouts**

Initiate sequential interactions by defining your own "Trigger" class as an environment. Whenever a specified pattern appears in the model’s output, LiteChainRL pauses generation, invokes the environment logic, and resumes—enabling complex, agent‑style workflows.

2. **LoRA/QLoRA Support**

Leverage low‑rank adapters to fine‑tune large models with a fraction of the memory footprint required for full‑parameter updates.

3. **Lightweight Design**

- Rollouts & Log-Probs: Powered by vLLM for blazing‑fast sample generation
- Training Backend: Pure PyTorch, no heavy scheduler or cluster required
- Rapid Prototyping: High‑level abstractions let you focus on ideas, not infrastructure

4. **Agent-Ready Abstraction**

The built‑in Trigger system lets you connect any callable environment—be it function execution, API calls, or custom simulator—to the model’s generation loop, making LiteChainRL an ideal playground for building interactive agents.

## Usage

Currently only support GRPO, more RL algorithms coming soon.
