import os
import os.path as osp
import re
import string
import sys
import traceback

import torch
from tqdm import trange

from consistency.Medalpaca.model_medalpaca import init_medalpaca_model
import pandas as pd


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import const
from arguments import args
from consistency.data_consistency import load_data_consistency, \
    load_results_consistency, get_consistency_results_path
from setup import project_setup
from consistency.Medalpaca.params_medalpaca import *

if osp.exists(const.HOME_DIR_LINUX):
    cuda_path = "/usr/local/cuda-11.7/bin/nvcc"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f"{cuda_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = cuda_path


def format_question(d):
    question = d["question"]
    options = d["options"]
    for k, v in options.items():
        question += f"\n{k}: {v}"
    return question


def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str

    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[
        start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[
        end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""


def starts_with_capital_letter(input_str):
    """
    The answers should start like this:
        'A: '
        'A. '
        'A '
    """
    pattern = r'^[A-Z](:|\.|) .+'
    return bool(re.match(pattern, input_str))


def run_consistency_medalpaca():
    path = get_consistency_results_path(args)

    model = init_medalpaca_model(args)

    sampling['temperature'] = args.temperature
    examples = load_data_consistency(args)

    results_df = load_results_consistency(args)

    def save():
        if osp.exists(path):
            with pd.ExcelWriter(path, mode='a', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=args.target_language, index=False)
        else:
            results_df.to_excel(path, sheet_name=args.target_language, index=False)

    idx_start = 0

    for idx_row in trange(idx_start, len(examples)):

        example = examples.iloc[idx_row]
        results_df.loc[idx_row, const.ID] = example[const.ID]

        if const.QUESTION in example:
            question = results_df.loc[
                idx_row, const.QUESTION] = example[
                const.QUESTION]

            if not isinstance(question, str):
                continue

        if args.target_language != "English":
            question_translated = results_df.loc[
                idx_row, const.QUESTION_TRANSLATED] = example[
                    const.QUESTION_TRANSLATED]

            if not isinstance(question_translated, str):
                continue

        for i in range(0, args.num_answers, args.batch_size):
            print("=" * 30)
            print(f"Example {idx_row}\tanswer {i}")

            input_questions = [question if args.target_language == 'English'
                               else question_translated] * args.batch_size

            print(f"> Question: {question if args.target_language == 'English' else question_translated}")

            if f"answer_{i}" in results_df.columns and isinstance(
                    results_df.loc[idx_row, f"answer_{i}"],
                    str):
                print(f"Skip Ex{idx_row}\tAns{i}")
                continue

            print('-' * 25)

            try:

                responses = model.batch_inference(
                    instruction=f"Answer this question in {args.target_language}.",
                    inputs=input_questions,
                    output="The answer to the question is:",
                    verbose=True,
                    **sampling
                )

                for j, response in enumerate(responses):
                    print(f"Answer: {response}")

                    results_df.loc[idx_row, f"answer_{i + j}"] = \
                        response
            except Exception as e:
                traceback.print_exc()

        if (idx_row + 1) % 10 == 0:
            print(f"Saving results to {path}...", end=" ")
            save()
            print("Done!")

    save()


if __name__ == "__main__":

    project_setup()
    os.makedirs(osp.join(args.output_dir, "consistency"), exist_ok=True)
    args.batch_size = 5

    for language in ["English", "Spanish", "Chinese", "Hindi"]:

        for temperature in [0.001, 0.25, 0.5, 0.75, 1.0]:
            args.target_language = language
            args.temperature = temperature
            run_consistency_medalpaca()

            torch.cuda.empty_cache()
