class BaseReward:
    def __init__(self, engine_instance):
        pass

    def reward(self, items, metadata):
        """
        Get the reward from the rollout generation and ground_truth.
        """
        texts = [item["text"] for item in items]
        raise NotImplementedError("Please implement reward method.")
