import os
import sys
import os.path as osp

import pandas as pd

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append("..")

import const
from dataloader import load_data


def load_data_consistency(args):
    if args.dataset_name == "liveqa":
        examples = load_data.load_LiveQA(language=args.target_language)

    elif args.dataset_name == "medicationqa":
        examples = load_data.load_MedicationQA(language=args.target_language)

    elif args.dataset_name == "healthqa":
        assert args.interval == 10
        examples = load_data.load_HealthQA(split=args.split,
                                 language=args.target_language)

    elif args.dataset_name == "mlecqa":

        examples = load_data.load_MLECQA(language=args.target_language)
        if args.split == const.DEV:
            assert len(examples) == 2205


    else:
        raise NotImplementedError

    return examples


def get_consistency_results_path(args):
    if args.model != "gpt35":
        model_prefix = f"{args.model}_"

    else:
        model_prefix = ""

    if args.dataset_name in ["liveqa", "medicationqa"]:
        path = osp.join(args.output_dir, "consistency",
                        f"{model_prefix}{args.dataset_name}_consistency_temp{args.temperature}_{args.target_language}.xlsx")

    elif args.dataset_name in ["healthqa", "mlecqa"]:
        path = osp.join(args.output_dir, "consistency",
                        f"{model_prefix}{args.dataset_name}_{args.split}_consistency_temp{args.temperature}_{args.target_language}.xlsx")

    else:
        raise NotImplementedError
    return path

def load_results_consistency(args):
    path = get_consistency_results_path(args)

    if osp.exists(path):
        results_df = pd.read_excel(path)

        print(f"Loaded {len(results_df)} examples from {path}")


    else:
        results_df = pd.DataFrame()

    return results_df
