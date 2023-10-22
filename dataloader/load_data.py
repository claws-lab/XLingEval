import json
import os
import sys
from os import path as osp

import pandas as pd

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import const


def load_HealthQA(split: str, language: str = 'English', task: str = "consistency"):
    print(f"Loading HealthQA with split {split} and Language {language} ...")

    if osp.basename(os.getcwd()) == "XLingHealth_Benchmark":
        path = "HealthQA.xlsx"

    else:
        path = osp.join("XLingHealth_Benchmark", "HealthQA.xlsx")

    raw_df = pd.read_excel(path, sheet_name=language)

    if task == "verifiability":
        return raw_df

    elif task in ["consistency", "accuracy"]:
        df = raw_df[raw_df["label"] == 1]
        return df

    else:
        raise ValueError(f"Unknown task {task}")




def load_MedicationQA(language: str = "English", task: str = "consistency"):

    if osp.basename(os.getcwd()) == "XLingHealth_Benchmark":
        path = "MedicationQA.xlsx"

    else:
        path = osp.join("XLingHealth_Benchmark", "MedicationQA.xlsx")

    raw_df = pd.read_excel(path, sheet_name=language)

    if task == "verifiability":
        raw_df["neg_sample"] = [[x[const.ID]] + eval(x["neg_sample"]) for _, x in raw_df.iterrows()]
        df = raw_df.explode("neg_sample")
        df.drop(const.ID, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # LiveQA does not provide negative samples, so we do negative sampling here.
        df[const.LABEL] = [1 if i % 5 == 0 else 0 for i in range(len(df))]
        df[const.ANSWER] = raw_df.loc[df["neg_sample"].values.astype(int), const.ANSWER].reset_index(drop=True)

        if language != "English":
            df[const.ANSWER_TRANSLATED] = raw_df.loc[df["neg_sample"].values.astype(int), const.ANSWER_TRANSLATED].reset_index(drop=True)


        return df

    else:
        return raw_df


def load_LiveQA(language="English", task: str = "consistency"):
    if osp.basename(os.getcwd()) == "XLingHealth_Benchmark":
        path = "LiveQA.xlsx"

    else:
        path = osp.join("XLingHealth_Benchmark", "LiveQA.xlsx")

    raw_df = pd.read_excel(path, sheet_name=language)

    if task == "verifiability":
        raw_df["neg_sample"] = [[x[const.ID]] + eval(x["neg_sample"]) for _, x in raw_df.iterrows()]
        df = raw_df.explode("neg_sample")
        df.drop(const.ID, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        # LiveQA does not provide negative samples, so we do negative sampling here.
        df[const.LABEL] = [1 if i % 5 == 0 else 0 for i in range(len(df))]
        df[const.ANSWER] = raw_df.loc[df["neg_sample"].values.astype(int), const.ANSWER].reset_index(drop=True)

        if language != "English":
            df[const.ANSWER_TRANSLATED] = raw_df.loc[df["neg_sample"].values.astype(int), const.ANSWER_TRANSLATED].reset_index(drop=True)

        return df

    else:
        return raw_df



def load_consistency_results(temp2lang_and_df: dict):
    """

    :param temp2lang_and_df:
    :return:
    """

    from arguments import args

    if temp2lang_and_df:
        print("Loading data from cache...")

    else:
        for temperature in const.TEMPERATURES:
            language2df = {}
            for language in const.LANGUAGES:
                args.language = language

                if args.dataset_name in ["healthqa"]:
                    filename = f"{args.dataset_name}_consistency_{args.split}_temp{float(temperature)}.xlsx"

                else:
                    filename = f"{args.dataset_name}_consistency_temp{temperature}.xlsx"

                df = pd.read_excel(
                    osp.join(args.output_dir, "summary", filename),
                    sheet_name=language, index_col=0)

                language2df[language] = df

            temp2lang_and_df[temperature] = language2df

    return temp2lang_and_df