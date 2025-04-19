class BaseReward:
    def __init__(self, engine_instance):
        pass

    def reward(self, items, metadata):
        """
        Get the reward from the rollout generation and ground_truth.

        :items: a list of `LoRAEngine.multi_turn_gen` result.
        :metadata: the third item wrapped in Dataset.

        :return: items append with their rewards (currently only support ORM).
        """
        texts = [item["text"] for item in items]
        raise NotImplementedError("Please implement reward method.")
