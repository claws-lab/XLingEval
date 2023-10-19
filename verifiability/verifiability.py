import os.path as osp
import traceback

import numpy as np
import pandas as pd

import const
from arguments import args
from dataloader.load_data import load_HealthQA, load_LiveQA, load_MedicationQA
from verifiability.prompts import prompt_verifiability
from setup import project_setup, openai_setup
from utils.utils_chatgpt import get_response
from utils.utils_misc import get_model_prefix, capitalize_and_strip_punctuation

project_setup(args)
openai_setup(args)
RETURN_EXPLANATION = False

results = {}





def run_verifiability(temperature: float, dataset_name: str, target_language: str):
    from utils.utils_misc import map_prediction_to_binary

    if dataset_name in ['healthqa']:

        path = osp.join(args.output_dir, "verifiability",
                        f"{get_model_prefix(args)}_{dataset_name}_verifiability_temp{temperature}_{args.split}"
                        f"_{target_language}.xlsx")

        examples = load_HealthQA(args.split, target_language)


    else:
        path = osp.join(args.output_dir, "verifiability",
                        f"{get_model_prefix(args)}{dataset_name}_verifiability_temp{temperature}.xlsx")

        if dataset_name in ['liveqa']:
            examples = load_LiveQA(target_language, task="verifiability")


        elif dataset_name in ['medicationqa']:
            examples = load_MedicationQA(target_language, task="verifiability")

        else:
            raise NotImplementedError

    def save():
        if osp.exists(path):
            with pd.ExcelWriter(path, mode='a', engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=target_language, index=False)
        else:
            results_df.to_excel(path, sheet_name=target_language, index=False)


    if osp.exists(path):
        results_df = pd.read_excel(path)
        print(f"Loaded {len(results_df)} examples from {path}")

    else:
        results_df = pd.DataFrame()
        results_df[const.PRED] = np.NaN
        results_df[const.ERROR] = np.NaN

    if args.fill_null_values:
        results_df[const.PRED] = results_df[const.PRED].apply(
            lambda x: map_prediction_to_binary(x, target_language))

    idx_start = 0 if args.fill_null_values else len(results_df)

    # Each row has a question and a sample answer
    for idx_row in range(idx_start, len(examples)):

        row = examples.loc[idx_row]

        # Copy the contents from the original data
        results_df.loc[idx_row, const.QUESTION] = row[const.QUESTION]
        results_df.loc[idx_row, const.ANSWER] = row[const.ANSWER]
        results_df.loc[idx_row, const.ID] = row.name
        results_df.loc[idx_row, const.LABEL] = row[const.LABEL]

        if args.fill_null_values:
            row_pred = results_df.iloc[idx_row]
            if row_pred[const.PRED] in ["Yes", "No"]:
                continue

        prompt = prompt_verifiability(
            row[const.QUESTION if target_language == "English" else const.QUESTION_TRANSLATED],
            row[const.ANSWER if target_language == "English" else
            const.ANSWER_TRANSLATED],
            target_language)

        print(f"{idx_row}\t{prompt}")

        try:
            response = get_response(prompt, temperature=temperature,
                                    deployment_id="gpt35")

            response = capitalize_and_strip_punctuation(response)

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            traceback.print_exc()
            results_df.loc[idx_row, const.ERROR] = str(e)
            response = np.NaN

        print(f"{idx_row}\t{response}")

        results_df.loc[idx_row, const.PRED] = response

        if (idx_row + 1) % 1000 == 0:
            save()

    save()

if __name__ == "__main__":

    for temperature in const.TEMPERATURES:
        for language in const.LANGUAGES:
            args.target_language = language
            args.temperature = temperature
            run_verifiability(dataset_name=args.dataset_name, temperature=temperature, target_language=language)
