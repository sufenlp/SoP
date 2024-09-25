import os
from src.utils.scorer import Scorer
from src.utils.utils import load_jsonl, write_jsonl


class ScoreCont:

    def __init__(self, base_pth, device="cuda:0"):
        self.scorer = Scorer(device=device)
        self.base_pth = base_pth

    def step(self, character_idx, beam_idx, step_idx):
        step_folder_pth = f"{self.base_pth}/character_{character_idx}/beam_{beam_idx}/step_{step_idx}/"
        next_step_folder_pth = f"{self.base_pth}/character_{character_idx}/beam_{beam_idx}/step_{step_idx+1}/"
        os.makedirs(next_step_folder_pth, exist_ok=True)

        opt_character_list = load_jsonl(step_folder_pth + "opt_character.jsonl")
        for opt_character in opt_character_list:
            input_pth = step_folder_pth + f"evaluation/{opt_character['gen_id']}.jsonl"
            asr = self.scorer.split_sentence_score(input_pth)
            opt_character["ASR"] = asr
        write_jsonl(opt_character_list, step_folder_pth + "opt_character.jsonl")

        # update the trajectory
        trajectory_list = load_jsonl(step_folder_pth + "trajectory.jsonl")
        trajectory_list_new = trajectory_list + opt_character_list
        trajectory_list_new = sorted(trajectory_list_new, key=lambda x: (x['ASR'], x['step'], x['gen_id']))
        write_jsonl(trajectory_list_new, next_step_folder_pth + "trajectory.jsonl")


        tmp_best_character = trajectory_list_new[-1]
        best_character = {"Character": character_idx,
                          "step": tmp_best_character['step'],
                          "gen_id": tmp_best_character['gen_id'],
                          "character_name": tmp_best_character['character_name'],
                          "character_description": tmp_best_character['character_description'],
                          "tmp_ASR": tmp_best_character['ASR']}
        good_step = False  # If the step is good, its optimize outcome would replace the top ones in the trajectory.
        # print(good_step)
        if step_idx == 0:
            good_step = True
            write_jsonl([1], step_folder_pth + '/good_step.jsonl')
        elif trajectory_list_new[-1]["ASR"] > trajectory_list[-1]['ASR']:
            good_step = True
            write_jsonl([1], step_folder_pth + '/good_step.jsonl')
        else:
            write_jsonl([0], step_folder_pth + '/good_step.jsonl')
        print("Is this step good?: ", good_step)
        return good_step, best_character





