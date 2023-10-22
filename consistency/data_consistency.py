import sys
import os.path as osp

import pandas as pd

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append("..")

from dataloader.load_data import load_LiveQA, load_MedicationQA, load_HealthQA


def load_data_consistency(args):
    if args.dataset_name == "liveqa":
        examples = load_LiveQA(language=args.target_language)

    elif args.dataset_name == "medicationqa":
        examples = load_MedicationQA(language=args.target_language)

    elif args.dataset_name == "healthqa":
        examples = load_HealthQA(split=args.split,
                                 language=args.target_language)

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
                        f"{model_prefix}{args.dataset_name}_consistency_temp{args.temperature}.xlsx")

    elif args.dataset_name in ["healthqa"]:
        path = osp.join(args.output_dir, "consistency",
                        f"{model_prefix}{args.dataset_name}_{args.split}_consistency_temp{args.temperature}.xlsx")

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
