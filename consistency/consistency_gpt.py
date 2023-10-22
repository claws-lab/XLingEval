"""Test consistency of the model.

Currently, we use the following steps:
* Ask ChatGPT a question and let it return the answers in bullet-by-bullet format.
* Compare the pairwise similarities between the answers.
    For each question, if N answers are generated, then there are C(N, 2) pairs of answers.

Previously, we used the following steps:
* Ask ChatGPT a question and let it return the answers in bullet-by-bullet format.
* For each answer
    * Ask ChatGPT if the answer is a valid response to the question.
    * [Optional] Modify the question to make it factually answer. Ask ChatGPT the question and the modified answer


Created: 2023.5.12
"""
import traceback

import numpy as np
from tqdm import trange
import sys
import os.path as osp

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


import const
from arguments import args
from consistency.data_consistency import load_data_consistency, \
    load_results_consistency, get_consistency_results_path
from consistency.prompts import prompt_consistency
from setup import project_setup, openai_setup
from utils.utils_chatgpt import get_response



def consistency(args):
    examples = load_data_consistency(args)

    results_df = load_results_consistency(args)

    path = get_consistency_results_path(args)

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

        # if idx_row < len(results_df) and not isinstance(example[field], (str, float)):
        #     results_df.loc[idx_row, field] = str(example[field])

        for i in range(args.num_answers):
            print(f"Example {idx_row}\tanswer {i}")
            if f"answer_{i}" in results_df.columns and isinstance(
                    results_df.loc[idx_row // args.interval, f"answer_{i}"],
                    str):
                print(f"Skip Ex{idx_row}\tAns{i}")
                continue

            try:
                prompt = prompt_consistency(question if (
                            args.target_language == "English" or args.dataset_name == "mlecqa") else question_translated,
                                                       language,
                                                       use_bullet_point=False)
                print(prompt)
                answer = get_response(prompt, temperature=args.temperature,
                                      deployment_id="gpt35")
                print(answer)
                results_df.loc[idx_row // args.interval, f"answer_{i}"] = answer


            except Exception as e:
                print(e)
                traceback.print_exc()
                results_df.loc[idx_row // args.interval, f"answer_{i}"] = np.NaN

                continue

        if (idx_row // args.interval) % 20 == 0 and i == args.num_answers - 1:
            results_df.to_excel(path, index=False)

    results_df.to_excel(path, index=False)


if __name__ == "__main__":

    project_setup()
    openai_setup(args)

    if args.dataset_name == "healthqa":
        args.interval = 10

    for language in ["English", "Chinese", "Hindi", "Spanish"]:

        for temperature in [0., 0.25, 0.5, 0.75, 1.0]:

            args.target_language = language
            args.temperature = temperature

            print("-" * 20)
            print(
                f"> Lang {args.target_language}, Temperature {args.temperature}")
            print("-" * 20)

            assert args.dataset_name in ["liveqa", "medicationqa", "healthqa"]

            consistency(args)
