import torch.multiprocessing as mp

from rollout.lora_engine import LoRAEngine
from reward.base_reward import BaseReward


def engine_worker(
    input_queue,
    control_event,
    output_queue,
    reward_class,
    **kwargs,
):
    lora_engine = LoRAEngine(
        **kwargs,
    )
    reward_instance = reward_class(lora_engine)

    control_event.wait()
    while True:
        try:
            requests = input_queue.get_nowait()
        except:
            control_event.wait()
            lora_engine.update_lora(kwargs["lora_path"])
            continue

        if requests == -1:
            break

        # requests are (prompt, env, ground_truth)
        responses = lora_engine.multi_turn_gen(requests[0], requests[1])
        responses = reward_instance.reward(responses, requests[2])
        output_queue.put(responses)


def engine_manager(
    dataset,
    devices_list,
    group_size,
    update_batch,
    update_event,
    mini_batch,
    mini_batch_queue,
    **kwargs,
):
    dataset = dataset()
    mp.set_start_method("spawn")

    input_queue = mp.Queue()
    output_queue = mp.Queue()
    kwargs["input_queue"] = input_queue
    kwargs["output_queue"] = output_queue

    events = []
    for idx in range(len(devices_list)):
        control_event = mp.Event()
        events.append(control_event)
        kwargs["control_event"] = control_event
        kwargs["cuda_visible_devices"] = devices_list[idx]
        kwargs["seed"] = idx + 42
        p = mp.Process(target=engine_worker, kwargs=kwargs)
        p.start()

    send_count = 0
    for prompt, env, ground_truth in dataset:
        prompts = [prompt for _ in range(group_size)]
        envs = [env.copy() for _ in range(group_size)]
        ground_truths = [ground_truth for _ in range(group_size)]
        input_queue.put((prompts, envs, ground_truths))
        send_count += 1

        if send_count * group_size >= update_batch:
            for e in events:
                e.set()
        else:
            continue

        output_items = []
        for _ in range(send_count):
            output_items.extend(output_queue.get())
            if len(output_items) >= mini_batch:
                mini_batch_queue.put(output_items[:mini_batch])
                output_items = output_items[mini_batch:]

        update_event.wait()
        send_count = 0
