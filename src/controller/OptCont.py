from src.component.optLM import OptLM
from src.component.template4opt import Template4OPT
from src.utils.utils import load_jsonl


class OptCont:

    def __init__(self,
                 base_pth, model_name, output_length,
                 gen_num=4, trajectory_len=4, random_character_first=True, use_exist=True,
                 max_n_tokens=2048, temperature=0.01, top_p=1,
                 loaded_model=None, loaded_tokenizer=None):
        self.optLM = OptLM(model_name=model_name,
                           max_n_tokens=max_n_tokens, temperature=temperature, top_p=top_p,
                           loaded_model=loaded_model, loaded_tokenizer=loaded_tokenizer)
        self.base_pth = base_pth
        self.output_length = output_length
        self.gen_num = gen_num
        self.trajectory_len = trajectory_len
        self.random_character_first = random_character_first
        self.use_exist = use_exist

    def step(self, character_idx, beam_idx, step_idx):
        beam_folder_pth = f"{self.base_pth}/character_{character_idx}/beam_{beam_idx}"
        step_folder_pth = beam_folder_pth + f"/step_{step_idx}"
        existed_character_list = load_jsonl(beam_folder_pth + "/existed_character.jsonl") if self.use_exist else []  #[1:]  # 在 Character n 的文件夹下有即将与其组合的 Existed Characters
        if self.trajectory_len:
            meta_character_list = load_jsonl(step_folder_pth + "/trajectory.jsonl")[-self.trajectory_len:]
        else:
            meta_character_list = []
        templator = Template4OPT(random_c=self.random_character_first)
        if not self.random_character_first:
            for idx in range(self.gen_num):
                print(f"Optimizing {idx + 1}/{self.gen_num}")
                gen_character = False
                while not gen_character:
                    opt_template = templator.get_template(meta_character_list=meta_character_list,
                                                          output_length=self.output_length,
                                                          existed_character_list=existed_character_list)
                    gen_character = self.optLM.get_response(prompt=opt_template,
                                                            step_idx=step_idx,
                                                            output_folder=step_folder_pth,
                                                            gen_idx=idx)

        else:
            for idx in range(self.gen_num):
                print(f"Optimizing {idx + 1}/{self.gen_num}")
                gen_character = False
                while not gen_character:
                    random_character = self.optLM.random_character(gen_idx=idx, output_folder=step_folder_pth)
                    print(random_character)
                    opt_template = templator.get_template(meta_character_list=meta_character_list,
                                                          output_length=self.output_length,
                                                          random_character=random_character,
                                                          existed_character_list=existed_character_list)
                    # print(opt_template)
                    gen_character = self.optLM.get_response(prompt=opt_template,
                                                            step_idx=step_idx,
                                                            output_folder=step_folder_pth,
                                                            from_character=random_character,
                                                            gen_idx=idx)
                    print(gen_character)

    def get_loaded(self):
        """
        如果 attacker 和 target 同时为 llama 或 vicuna，则可以共用相同的模型参数来节省显存
        """
        return self.optLM.model.model, self.optLM.model.tokenizer


