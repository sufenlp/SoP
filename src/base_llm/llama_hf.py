import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import common
from src.base_llm.language_model import LanguageModel
from src.utils.config import LLAMA_PATH
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        each_step = torch.tensor([self.stop(input_id) for input_id in input_ids])
        if each_step.all():  # 全为 true 时 返回 true
            return True
        else:
            return False

    @staticmethod
    def stop(input_id):
        # stop 则返回 True
        if len(input_id) > 20:
            # 输出 20个相同token 或 正常结束 2: </s>
            return torch.eq(input_id[-20:], input_id[-20]).all() or torch.tensor(2 in input_id)
        else:
            return torch.tensor(False)


class LlamaHF(LanguageModel):

    def __init__(self, model_name="llama-2", loaded_model=None, loaded_tokenizer=None):
        os.environ["CUDA_DEVICE_ORDER"] = "2,3"
        self.model_name = model_name
        self.stopping_criteria = StoppingCriteriaList([StopOnTokens()])

        if loaded_model:
            self.model = loaded_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(LLAMA_PATH, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="balanced").eval()

        if loaded_tokenizer:
            self.tokenizer = loaded_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.tokenizer.padding_side = 'left'
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self,
                         full_prompts_list,
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}

        output_ids = None
        finish = False
        while not finish:
            try:
                # Batch generation
                if temperature > 0:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_n_tokens,
                        do_sample=True,
                        temperature=temperature,
                        eos_token_id=self.eos_token_ids,
                        top_p=top_p,
                        stopping_criteria=self.stopping_criteria
                    )
                else:
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_n_tokens,
                        do_sample=False,
                        eos_token_id=self.eos_token_ids,
                        top_p=1,
                        temperature=1,  # To prevent warning messages
                        stopping_criteria=self.stopping_criteria
                    )
                finish = True
            except Exception as e:
                print(e)
                continue

        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # with open("/nfsshare/home/xuhaochen/yy/MoreCharacter/workspace/look.txt", "a+") as f:
        #     print(output_ids.cpu().numpy())
        #     f.write(str(output_ids.cpu().numpy()))
        #     for i in output_ids:
        #         print(i.cpu().numpy())
        #         f.write(str(i.cpu().numpy()))

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def generate(self,
                 full_prompt,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        outputs_list = self.batched_generate([full_prompt], max_n_tokens, temperature, top_p)
        return outputs_list[0]

    def extend_eos_tokens(self):
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1], 29913, 9092, 16675])

    @staticmethod
    def preprocess_input(prompt_list, system_prompt=None):
        input_list = []
        for prompt in prompt_list:
            conv = common.conv_template("llama-2")
            if system_prompt:
                conv.set_system_message(system_prompt)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            input_list.append(conv.get_prompt())
        return input_list


if __name__ == '__main__':
    llm = LlamaHF()
    input_list = LlamaHF.preprocess_input(["Write a story about making a cake."], "")
    print(input_list)
    print(llm.batched_generate(input_list, 512, 0, 0.95))

