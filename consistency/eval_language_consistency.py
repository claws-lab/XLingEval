import os.path as osp

import numpy as np
import pandas as pd

import const
from arguments import args
from eval.language_consistency import split_multilingual_paragraph, \
    compute_language_percentage
from setup import project_setup, openai_setup
import warnings
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel*")

def language_consistency():

    if args.dataset_name in ["liveqa", "medicationqa"]:

        path = osp.join(args.output_dir, "consistency",
                        f"{prefix}{args.dataset_name}_consistency_temp{args.temperature}_{args.source_language}.xlsx")

        path_results = osp.join(args.output_dir, "summary",
                                f"{prefix}{args.dataset_name}_language_consistency.xlsx")

    elif args.dataset_name in ["healthqa"]:

        path = osp.join(args.output_dir, "consistency",
                        f"{prefix}{args.dataset_name}_{args.split}_language_consistency_temp{args.temperature}_{args.source_language}.xlsx")

        path_results = osp.join(args.output_dir, "summary",
                                f"{prefix}{args.dataset_name}_language_consistency_{args.split}.xlsx")



    else:
        raise NotImplementedError

    if not osp.exists(path):
        print(f"Error: Not found ({path})")
        return

    df = pd.read_excel(path)


    print(f"Loaded {len(df)} examples from {path}")


    df_language_percentage = df.copy()

    def f(answer: str):
        if not isinstance(answer, str) or answer == "":
            return np.NaN


        sentences = split_multilingual_paragraph(answer)

        language_percentages = compute_language_percentage(sentences)

        return language_percentages

    for idx_answer in range(10):
        df_language_percentage[f"answer_{idx_answer}"] = df_language_percentage[f"answer_{idx_answer}"].apply(f)

    print(df_language_percentage)

    if osp.exists(path_results):
        print(f"FILE EXISTS. Appending ...")
        with pd.ExcelWriter(path_results, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df_language_percentage.to_excel(writer, sheet_name=f"{args.source_language}_temp{args.temperature}", index=False)

    else:
        print(f"Creating {path_results}")
        with pd.ExcelWriter(path_results, mode='w', engine='openpyxl') as writer:
            df_language_percentage.to_excel(writer, sheet_name=f"{args.source_language}_temp{args.temperature}", index=False)




    print("Done!")


def summarize_language_consistency(summary_df: pd.DataFrame):
    prefix = "" if args.target_language == "English" else "TRANSLATED_"
    prefix += "" if args.model == "gpt35" else f"{args.model}_"

    print(f"Prefix of this experiment: {prefix}")

    if args.dataset_name in ["liveqa", "medicationqa"]:

        path_results = osp.join(args.output_dir, "summary",
                                f"{prefix}{args.dataset_name}_language_consistency.xlsx")


    elif args.dataset_name in ["healthqa"]:

        path_results = osp.join(args.output_dir, "summary",
                                f"{prefix}{args.dataset_name}_language_consistency_{args.split}.xlsx")

    else:
        raise NotImplementedError


    df_language_percentage = pd.read_excel(path_results, sheet_name=f"{args.source_language}_temp{args.temperature}")

    def average_language_percentages_per_answer(row):

        language_distribution_per_answer_df = []
        for idx_answer in range(10):
            if isinstance(row[f'answer_{idx_answer}'], str):
                lang_distribution = eval(row[f'answer_{idx_answer}'])

                for lang in ["en", "es", "zh", "hi"]:
                    if lang not in lang_distribution:
                        lang_distribution[lang] = 0.

                lang_distribution = pd.Series(lang_distribution)
                if lang_distribution.sum() > 99.9:
                    language_distribution_per_answer_df += [lang_distribution]

                elif lang_distribution.sum() < 0.1:
                    continue

                else:
                    print(lang_distribution)

        if language_distribution_per_answer_df != []:


            language_percentages_per_example = pd.concat(language_distribution_per_answer_df, axis=1)

            language_percentages_per_example.fillna(0, inplace=True)

            language_percentages_per_example = language_percentages_per_example.mean(axis=1)

            language_percentages_per_example[const.ID] = row[const.ID]


            return language_percentages_per_example

        else:
            return np.NaN

    rows = []

    for idx_row in range(len(df_language_percentage)):
        row = df_language_percentage.iloc[idx_row]
        result = average_language_percentages_per_answer(row)
        if isinstance(result, pd.Series):
            rows += [result]

    language_consistency_summary_df = pd.concat(rows, axis=1).T.astype(
        {const.ID: int})

    print(f"Temp {args.temperature}\tLanguage {args.source_language}\tLoaded {len(df_language_percentage)} examples")

    return language_consistency_summary_df





if __name__ == "__main__":
    project_setup(args)
    openai_setup(args)

    # for language in ["Chinese"]:

    TEMPERATURES = [0.0, 1.0]

    prefix = "" if args.target_language == "English" else "TRANSLATED_"
    prefix += "" if args.model == "gpt35" else f"{args.model}_"

    print(f"Prefix of this experiment: {prefix}")

    # Uncomment these lines to calculate %language in each answer

    # for temperature in TEMPERATURES:
    #     for language in ["English", "Chinese", "Hindi", "Spanish"]:
    #         args.temperature = temperature
    #         args.source_language = language
    #         language_consistency()

    summary_df = pd.DataFrame(columns=[const.ID, const.LANGUAGE, const.TEMPERATURE, "en", "es", "zh", "hi"])

    path_summary = osp.join(args.output_dir, "summary",
                            f"SUMMARY_{prefix}{args.dataset_name}_language_consistency.xlsx")

    for temperature in TEMPERATURES:
        for language in ["Chinese", "English", "Hindi", "Spanish"]:
            args.temperature = temperature
            args.source_language = language
            language_consistency_summary_df = summarize_language_consistency(summary_df)

            if osp.exists(path_summary):
                print(f"FILE EXISTS. Appending ...")
                with pd.ExcelWriter(path_summary, mode='a',
                                        if_sheet_exists='replace') as writer:
                    language_consistency_summary_df.to_excel(writer,
                                                             sheet_name=f"{args.source_language}_temp{args.temperature}",
                                                             index=False)

            else:
                print(f"Creating {path_summary}")
                with pd.ExcelWriter(path_summary, mode='w', engine='openpyxl') as writer:

                    language_consistency_summary_df.to_excel(writer,
                                                             sheet_name=f"{args.source_language}_temp{args.temperature}",
                                                             index=False)


            language_consistency_summary_row = language_consistency_summary_df.mean(
                axis=0)
            language_consistency_summary_row[
                const.TEMPERATURE] = args.temperature
            language_consistency_summary_row[
                const.LANGUAGE] = args.source_language
            summary_df = summary_df.append(language_consistency_summary_row,
                                           ignore_index=True)

    with pd.ExcelWriter(path_summary, mode='a',
                                if_sheet_exists='replace', engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name=f"summary", index=False)
