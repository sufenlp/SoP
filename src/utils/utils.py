# *coding:utf-8 *
import jsonlines
import json
import os


def get_keys(file_pth) -> list:
    with open(file_pth, 'r') as f_in:
        tmp = f_in.read().splitlines()
        keys = [i.split('--')[-1] for i in tmp]
    return keys


def load_jsonl(data_pth):
    """
    load a jsonlines file to a list
    """
    dict_list = []
    with open(data_pth, 'r') as f_in:
        for item in jsonlines.Reader(f_in):
            dict_list.append(item)
    return dict_list


def write_jsonl(json_data_list, output_pth, mode='w+'):
    with open(output_pth, mode=mode) as f_out:
        for json_data in json_data_list:
            f_out.write(json.dumps(json_data) + '\n')


def get_key_list(file_pth) -> list:
    with open(file_pth, 'r') as f_in:
        tmp = f_in.read().splitlines()
        key_list = [i.split('--')[-1] for i in tmp]
    return key_list


def score_summary(folder_pth):
    file_list = os.listdir(folder_pth)

    score_summary_list = []
    for file in file_list:
        if '_score.jsonl' not in file:
            continue
        else:
            file_pth = folder_pth + '/' + file
            data = load_data(file_pth)
            asr_sum = 0
            for tmp in data:
                asr_sum += tmp['success']
            asr = asr_sum / len(data)
            asr = int(asr * 100 + 0.5)
            score_summary_list.append({"file": file, "asr": asr})
    write_jsonl(score_summary_list, folder_pth + '/' + 'score_summary.jsonl')


def find_hard_mal_inst(file_pth, output_pth):
    data = load_data(file_pth)




if __name__ == '__main__':
    # score_summary('../../output_new/PAIR/character_1/step_3')
    score_summary('../../output_new/PAIR/character_2/trajectory')
    # score_summary('../../output/gptfuzzer')