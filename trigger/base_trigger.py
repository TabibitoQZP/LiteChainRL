class BaseTrigger:
    def trigger(self, input_text: str):
        """
        Detect whether LLM sequence should stop. If so, return the edited input and specific output.
        For example, if you want the model to interatctly run the code, you can trigger when detect
        '```python```' wrapped code at the end. Then execute the code and get the output.

        :input_text: sequence from LLM
        :return: the output text
        """
        raise NotImplementedError("This method is not implemented.")

    def copy(self):
        """
        Copy a same Trigger of this instance.

        :return: a instance of Trigger
        """
        raise NotImplementedError("This method is not implemented.")
