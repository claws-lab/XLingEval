import os
import os.path as osp
import sys
import traceback

import torch
from tqdm import trange

from medalpaca.model_medalpaca import init_medalpaca_model

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import const
from arguments import args
from consistency.data_consistency import load_data_consistency, \
    load_results_consistency, get_consistency_results_path
from setup import project_setup

if osp.exists(const.HOME_DIR_LINUX):
    cuda_path = "/usr/local/cuda-11.7/bin/nvcc"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f"{cuda_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = cuda_path

import re
import string

from medalpaca.params_medalpaca import *


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


def consistency_medalpaca():
    path = get_consistency_results_path(args)

    model = init_medalpaca_model(args)

    sampling['temperature'] = args.temperature
    examples = load_data_consistency(args)

    results_df = load_results_consistency(args)

    idx_start = 0 if args.fill_null_values else len(results_df) * args.interval

    for idx_row in trange(idx_start, len(examples), args.interval):

        example = examples.iloc[idx_row]
        results_df.loc[idx_row // args.interval, const.ID] = example[const.ID]

        if args.dataset_name in ["healthqa", "liveqa", "medicationqa"]:

            if const.QUESTION in example:
                question = results_df.loc[
                    idx_row // args.interval, const.QUESTION] = example[
                    const.QUESTION]

                if not isinstance(question, str):
                    continue

            if args.target_language != "English":
                question_translated = results_df.loc[
                    idx_row // args.interval, const.QUESTION_TRANSLATED] = \
                    example[
                        const.QUESTION_TRANSLATED]

                if not isinstance(question_translated, str):
                    continue

        elif args.dataset_name == "mlecqa":

            if const.QTEXT in example:
                question = results_df.loc[
                    idx_row // args.interval, const.QUESTION] = example[
                    const.QTEXT]

                if not isinstance(question, str):
                    continue

        for i in range(0, args.num_answers, args.batch_size):
            print("=" * 30)
            print(f"Example {idx_row}\tanswer {i}")

            input_questions = [question if args.target_language == 'English'
                               else question_translated] * args.batch_size

            print(f"> Question: {question if args.target_language == 'English' else question_translated}")

            if f"answer_{i}" in results_df.columns and isinstance(
                    results_df.loc[idx_row // args.interval, f"answer_{i}"],
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

                    results_df.loc[idx_row // args.interval, f"answer_{i + j}"] = \
                        response
            except Exception as e:
                traceback.print_exc()

        if (idx_row // args.interval) % 1 == 0:
            print(f"Saving results to {path}...", end=" ")
            results_df.to_excel(path, index=False)
            print("Done!")

    results_df.to_excel(path, index=False)


if __name__ == "__main__":

    project_setup(args)

    assert args.dataset_name in ["liveqa"]

    if args.dataset_name == "healthqa":
        args.interval = 10

    for language in const.LANGUAGES:

        for temperature in const.TEMPERATURES:
            args.target_language = language
            args.temperature = temperature
            consistency_medalpaca()

            torch.cuda.empty_cache()
