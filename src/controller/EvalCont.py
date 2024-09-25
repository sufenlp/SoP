import os
import copy
import time
from src.component.evalLM import EvalLM
from src.component.template4eval import Template4EVAL
from src.utils.utils import load_jsonl, write_jsonl


class EvalCont:

    def __init__(self,
                 base_pth, dataset, model_name, temp_name="new_4", defense=None,
                 eval_batch=5, gen_num=3, system_prompt="",
                 max_n_tokens=4096, temperature=0.0, top_p=0.95,
                 loaded_model=None, loaded_tokenizer=None):
        self.evalLM = EvalLM(model_name,
                             max_n_tokens=max_n_tokens, temperature=temperature, top_p=top_p,
                             loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer, defense=defense)
        self.base_pth = base_pth
        self.dataset = dataset
        self.temp_name = temp_name
        self.eval_batch = eval_batch
        self.gen_num = gen_num
        self.system_prompt = system_prompt

        target = self.evalLM.model.model_name
        if "llama" in target:
            self.target_name = "LLaMA"
        elif "gpt" in target:
            self.target_name = "ChatGPT"
        elif "vicuna" in target:
            self.target_name = "Vicuna"

    def step(self, character_idx, beam_idx, step_idx, start_gen_idx=0):
        """
        For Character i, step n, evaluating the output of OptLM
        :param character_idx: Character idx: i
        :param beam_idx: n
        :param step_idx:Step idx: n
        :param start_gen_idx: n
        """
        print(f"---Working in {self.base_pth}")
        beam_folder_pth = f"{self.base_pth}/character_{character_idx}/beam_{beam_idx}"
        step_folder_pth = beam_folder_pth + f"/step_{step_idx}"
        evaluation_pth = step_folder_pth + "/evaluation"
        os.makedirs(evaluation_pth, exist_ok=True)
        existed_character_list = load_jsonl(beam_folder_pth + "/existed_character.jsonl")  # 在 Character n 的文件夹下有即将与其组合的 Existed Characters
        opt_character_list = load_jsonl(step_folder_pth + "/opt_character.jsonl")
        for opt_character in opt_character_list:
            gen_idx = opt_character["gen_id"]
            if gen_idx < start_gen_idx:
                continue
            tmp_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"---{tmp_time}---Evaluating character-{character_idx} beam-{beam_idx} step-{step_idx} gen_id-{gen_idx}")
            used_character_list = copy.deepcopy(existed_character_list)  # 此处要深拷贝
            used_character_list.append({"character_name": opt_character["character_name"],
                                        "character_description": opt_character["character_description"]})
            # print(used_character_list)
            used_character_name_list = [used_character['character_name'] for used_character in used_character_list]  # 记录每条数据使用的 character name 用于后续分割
            templator = Template4EVAL(used_character_list=used_character_list, temp_name=self.temp_name, targetlm_name=self.target_name)
            input_template = templator.get_origin_template()
            output_pth = evaluation_pth + f"/{opt_character['gen_id']}.jsonl"
            self.evalLM.get_response(prompt=input_template,
                                     dataset=self.dataset,
                                     output_pth=output_pth,
                                     used_character_name_list=used_character_name_list,
                                     batch_size=self.eval_batch,
                                     gen_num=self.gen_num,
                                     system_prompt=self.system_prompt)

    def get_loaded(self):
        """
        如果 attacker 和 target 同时为 llama 或 vicuna，则可以共用相同的模型参数来节省显存
        """
        return self.evalLM.model.model, self.evalLM.model.tokenizer


