
import os
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
import pandas as pd


def run_consistency(dataset_name: str, target_language: str, temperature: float):
    examples = load_data_consistency(args)

    results_df = load_results_consistency(args)

    path = get_consistency_results_path(args)

    idx_start = 0

    def save():
        if osp.exists(path):
            with pd.ExcelWriter(path, mode='a', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=target_language, index=False)
        else:
            results_df.to_excel(path, sheet_name=target_language, index=False)



    for idx_row in trange(idx_start, len(examples)):
        example = examples.iloc[idx_row]
        results_df.loc[idx_row, const.ID] = example[const.ID]

        question = results_df.loc[
            idx_row, const.QUESTION] = example[const.QUESTION]

        if target_language != "English":
            question_translated = results_df.loc[
                idx_row, const.QUESTION_TRANSLATED] = \
            example[
                const.QUESTION_TRANSLATED]

            if not isinstance(question_translated, str):
                continue

        for i in range(args.num_answers):
            print(f"Example {idx_row}\tanswer {i}")
            if f"answer_{i}" in results_df.columns and isinstance(
                    results_df.loc[idx_row, f"answer_{i}"],
                    str):
                print(f"Skip Ex{idx_row}\tAns{i}")
                continue

            try:
                prompt = prompt_consistency(question if (
                            target_language == "English" or args.dataset_name == "mlecqa") else question_translated,
                                                       language,
                                                       use_bullet_point=False)
                print(prompt)
                answer = get_response(prompt, temperature=temperature,
                                      deployment_id="gpt35")
                print(answer)
                results_df.loc[idx_row, f"answer_{i}"] = answer


            except Exception as e:
                print(e)
                traceback.print_exc()
                results_df.loc[idx_row, f"answer_{i}"] = np.NaN

                continue

        if idx_row % 20 == 0:
            save()

            results_df.to_excel(path, index=False)

    results_df.to_excel(path, index=False)


if __name__ == "__main__":

    project_setup()
    openai_setup(args)

    os.makedirs(osp.join(args.output_dir, "consistency"), exist_ok=True)

    for language in ["Spanish", "English", "Chinese", "Hindi"]:

        for temperature in [0., 0.25, 0.5, 0.75, 1.0]:

            args.target_language = language
            args.temperature = temperature

            print("-" * 20)
            print(
                f"> Lang {args.target_language}, Temperature {args.temperature}")
            print("-" * 20)

            assert args.dataset_name in ["liveqa", "medicationqa", "healthqa"]

            run_consistency(args.dataset_name, language, temperature)
