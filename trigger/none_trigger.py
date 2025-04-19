from trigger.base_trigger import BaseTrigger


class NoneTrigger(BaseTrigger):
    def trigger(self, input_text: str):
        return None

    def copy(self):
        return NoneTrigger()
