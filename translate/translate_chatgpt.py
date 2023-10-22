import json
import os.path as osp
import time
import traceback
from pprint import pprint

import numpy as np
import openai
import pandas as pd
from googletrans import Translator

from openai import APIError

import const
from arguments import args


from translate.prompts import prompt_translate
from verifiability.setup import project_setup, openai_setup
from utils.utils_chatgpt import get_response
from utils.utils_misc import get_model_prefix

project_setup()
openai_setup(args)

def translate_answers():

    assert args.source_language != "English"
    assert args.target_language == "English"


    prefix_translated = "TRANSLATED_"
    prefix_translated += get_model_prefix(args)

    if args.dataset_name in ["healthqa", "mlecqa"]:




        path_untranslated = osp.join(args.output_dir, "consistency",
                    f"{get_model_prefix(args)}{args.dataset_name}_{args.split}_consistency_temp{args.temperature}_{args.source_language}.xlsx")



        path_translated = osp.join(args.output_dir, "consistency",
                                     f"{prefix_translated}{args.dataset_name}_{args.split}_consistency_temp{args.temperature}_{args.source_language}.xlsx")

    elif args.dataset_name in ["liveqa", "medicationqa"]:
        path_untranslated = osp.join(args.output_dir, "consistency",
                   f"{get_model_prefix(args)}{args.dataset_name}_consistency_temp{args.temperature}_{args.source_language}.xlsx")
        path_translated = osp.join(args.output_dir, "consistency",
                   f"{prefix_translated}{args.dataset_name}_consistency_temp{args.temperature}_{args.source_language}.xlsx")

    else:
        raise NotImplementedError

    if osp.exists(path_untranslated):
        df = pd.read_excel(path_untranslated)

    else:
        print(f"\tNOT exist: {path_untranslated}")
        return

    if osp.exists(path_translated):
        results_df = pd.read_excel(path_translated)

    else:
        results_df = pd.DataFrame(columns=df.columns)

    idx_start = 0 if args.fill_null_values else len(results_df)



    for idx_row in range(idx_start, len(df)):
        # if args.dataset_name in ["healthqa", "liveqa", "medicationqa"]:
        #     results_df.loc[idx_row, const.QUESTION] = df.loc[idx_row, const.QUESTION]
        #
        # elif args.dataset_name in ["mlecqa"]:
        #
        #     results_df.loc[idx_row, const.QTEXT] = df.loc[
        #         idx_row, const.QTEXT]
        #
        # else:
        #     raise NotImplementedError

        print(df.loc[idx_row, const.ID])
        results_df.loc[idx_row, const.ID] = df.loc[idx_row, const.ID]
        results_df.loc[idx_row, const.QUESTION] = df.loc[idx_row, const.QUESTION]

        for i in range(args.num_answers):
            if f"answer_{i}" in df.columns and isinstance(df.loc[idx_row, f"answer_{i}"], str):

                answer = df.loc[idx_row, f"answer_{i}"]

                if args.fill_null_values and isinstance(results_df.loc[idx_row, f"answer_{i}"], str):
                    continue
                USE_GOOGLE_TRANSLATE = True
                if not USE_GOOGLE_TRANSLATE:
                    try:
                        prompt = prompt_translate(answer, args.source_language,
                                                  args.target_language)

                        answer_translated = get_response(prompt, temperature=0.0,
                                                         deployment_id="gpt35")
                        if isinstance(answer_translated, str):
                            answer_translated = answer_translated.replace("```", "")
                        results_df.loc[idx_row, f"answer_{i}"] = answer_translated

                    except Exception as e:
                        traceback.print_exc()
                        pass
                else:
                    try:
                        answer_translated = translator.translate(answer,
                                                                 src="auto",
                                                                 dest=
                                                                 const.LANG2SHORT[
                                                                     args.target_language]).text

                        results_df.loc[
                            idx_row, f"answer_{i}"] = answer_translated

                    except Exception as e:
                        traceback.print_exc()
                        results_df.loc[idx_row, const.ERROR] = str(e)
                        results_df.loc[idx_row, f"answer_{i}"] = np.NaN

                print(f"A {idx_row}-{i}\t{results_df.loc[idx_row, f'answer_{i}']}")

        if idx_row % 50 == 0:
            results_df.to_excel(path_translated, index=False)

    results_df.to_excel(path_translated, index=False)




if __name__ == "__main__":
    translator = Translator()

    for dataset_name in ["healthqa"]:
        # for language in ["Chinese", "Hindi", "Spanish"]:
        for language in ["Spanish",]:

            for temperature in [0.0]:

                args.dataset_name = dataset_name
                args.source_language = language
                args.temperature = temperature
                # Translate answers generated by GPT from the question
                assert args.source_language != "English"
                assert args.target_language == "English"

                print("-" * 20)
                print(f"> Lang {args.source_language}, Temperature {args.temperature}")
                print("-" * 20)

                translate_answers()


