import os
import pandas as pd
from typing import Dict, List
import jsonlines


def load_jsonl(pth, tag):
    output_list = []
    with open(pth, 'r') as f_in:
        for item in jsonlines.Reader(f_in):
            output_list.append(item[tag])
    return output_list


def load_data(dataset, dataset_size, start_idx=0, end_idx=None) -> List[Dict]:
    if dataset == "advbench_behaviors_custom":
        df = pd.read_csv('data/advbench/harmful_behaviors_custom.csv')
        dataset = list(df["goal"][:dataset_size])
    if dataset == "advbench_behaviors_custom_filtered":
        df = pd.read_csv('data/advbench/advbench_filtered.csv')
        dataset = list(df["goal"][:dataset_size])
    elif dataset == "advbench_behaviors_custom_complementary":
        df = pd.read_csv('data/advbench/harmful_behaviors_custom.csv')
        filter_index = list(df["Original index"])
        df = pd.read_csv('data/advbench/harmful_behaviors.csv')
        harmful_behaviors = list(df["goal"][:dataset_size])
        dataset = [harmful_behaviors[i] for i in range(len(harmful_behaviors)) if i not in filter_index]
    elif dataset == "gptfuzzer":
        df = pd.read_csv('data/gptfuzzer/question_list.csv')
        dataset = list(df["text"][start_idx:dataset_size])
    elif "morse" in dataset:
        dataset = load_jsonl(dataset, "goal")[start_idx:dataset_size]
    else:
        df = pd.read_csv(dataset)
        dataset = list(df["goal"][start_idx:dataset_size])
    return dataset
