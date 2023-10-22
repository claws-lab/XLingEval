import os
import os.path as osp
import traceback

import numpy as np
import pandas as pd
from tqdm import trange

import const
import const_verifiability
from arguments import args
from dataloader.load_data import load_HealthQA, load_LiveQA, load_MedicationQA
from verifiability.Medalpaca.model_medalpaca import init_medalpaca_model
from verifiability.prompts import prompt_verifiability
from verifiability.setup import project_setup, openai_setup
from utils.utils_chatgpt import get_response
from utils.utils_misc import get_model_prefix, capitalize_and_strip_punctuation
from verifiability.Medalpaca.params_medalpaca import *


project_setup()
openai_setup(args)
RETURN_EXPLANATION = False

results = {}



def run_verifiability(temperature: float, dataset_name: str, target_language: str):
    from utils.utils_misc import map_prediction_to_binary

    os.makedirs(osp.join(args.output_dir, "verifiability"), exist_ok=True)

    if dataset_name in ['healthqa']:

        path = osp.join(args.output_dir, "verifiability",
                        f"{get_model_prefix(args)}{dataset_name}_verifiability_temp{temperature}_{args.split}"
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

    idx_start = 0

    def format_question(question, answer):
        return f"Question: {question}\nResponse: {answer}"


    if args.model.startswith("medalpaca"):

        questions = examples[const.QUESTION if
        args.target_language == "English" else const.QUESTION_TRANSLATED].tolist()

        answers = examples[const.ANSWER if
        args.target_language == "English" else const.ANSWER_TRANSLATED].tolist()

        ids = examples[const.ID].values

        input_questions = [format_question(question, answer) for question,
        answer in
                           zip(questions, answers)]

        sampling['temperature'] = args.temperature

        results_df[const.ID] = ids
        results_df[const.QUESTION] = None
        results_df[const.ANSWER] = None

        for idx_row in trange(idx_start, len(input_questions), args.batch_size):

            results_df.loc[idx_row:idx_row + args.batch_size - 1, const.QUESTION] \
                = questions[idx_row:idx_row + args.batch_size]
            results_df.loc[idx_row:idx_row + args.batch_size - 1, const.ANSWER] = \
                answers[idx_row:idx_row + args.batch_size]

            try:
                batch = input_questions[idx_row:idx_row + args.batch_size]
                responses = model.batch_inference(
                    instruction=f"Answer me 'Yes' or 'No'.",
                    inputs=batch,
                    output="The answer to the question is:",
                    verbose=True,
                    **sampling
                )

            except Exception as e:
                traceback.print_exc()
                continue

            results_df.loc[idx_row:idx_row + args.batch_size - 1, const.PRED] = responses

            if (idx_row % 20 == 0 or idx_row == len(liveqa_examples) - 1):
                print(f"Saving results to {path}...", end=" ")
                # results_df.reset_index(drop=True).drop("Unnamed: 0", axis=1, errors="ignore").to_excel(path, index=False)

                results_df.to_excel(path, index=False)
                print("Done")


    else:

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

            if (idx_row + 1) % 20 == 0:
                save()
        save()

if __name__ == "__main__":

    if args.model.startswith("medalpaca"):
        model = init_medalpaca_model(args)

    elif args.model.startswith("gpt"):
        from utils.utils_chatgpt import get_response

    else:
        model = None

    for temperature in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for language in ["Spanish", "English", "Chinese", "Hindi"]:
            args.target_language = language
            args.temperature = temperature
            run_verifiability(dataset_name=args.dataset_name, temperature=temperature, target_language=language)
