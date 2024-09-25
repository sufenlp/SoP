class LanguageModel:
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError

    @staticmethod
    def preprocess_input(prompt_list, system_prompt=None):
        if system_prompt:
            return [{"system_prompt": system_prompt, "prompt": prompt} for prompt in prompt_list]
        else:
            return [{"prompt": prompt} for prompt in prompt_list]
