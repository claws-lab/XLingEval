import os.path as osp
import string

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, \
    roc_auc_score, confusion_matrix

import const
from arguments import args
from setup import project_setup
from utils.utils_misc import map_prediction_to_binary, map_prediction_to_binary_MedAlpaca

project_setup(args)

ANSWER_MAP = {
    'Entailment': 1,
    'Not entailment': 0,
    "Yes": 1,
    "No": 0,
    np.NaN: -1,
}


def plot_confusion_matrix(df: pd.DataFrame, language: str, pos_label: int,
                          args):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Compute confusion matrix

    # If labels is [1, 0], we calculate the confusion matrix for class 1 as the positive label
    cm = confusion_matrix(df[const.LABEL], df[const.PRED_BINARY],
                          labels=[1, 0] if pos_label == 1 else [0, 1])

    # Create a dataframe from the confusion matrix.
    cm_df = pd.DataFrame(cm,
                         index=['Actual Positive', 'Actual Negative'],
                         columns=['Predicted Positive', 'Predicted Negative'])

    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.6)

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

    if args.dataset_name in ["healthqa", "mlecqa"]:
        name = f"{args.dataset_name}_{args.split}_{language}_pos-label={pos_label}"
        title = f"{args.dataset_name}, Lang:{language}, Split:{args.split}, Positive Label:{pos_label}"

    else:
        name = f"{args.dataset_name}_{language}_pos-label={pos_label}"
        title = f"{args.dataset_name}, Lang:{language}, Positive Label:{pos_label}"

    plt.title(f'{title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.savefig(f"{name}.png", dpi=300)


def assign_binary_label_verifiability(x):
    if not isinstance(x, str):
        return x
    if "not entailment" in x.lower():
        return "Not entailment"
    elif "entailment" in x:
        return "Entailment"
    else:
        return x


def convert_prediction_to_binary(df: pd.DataFrame, model_name: str):
    df[const.PRED] = df[const.PRED].str.capitalize()
    df[const.PRED] = df[const.PRED].str.strip(string.punctuation)
    print(df[const.PRED].value_counts(dropna=False))

    if model_name in ["gpt35", "gpt4"]:
        f = map_prediction_to_binary

    elif model_name.startswith("medalpaca"):
        f = map_prediction_to_binary_MedAlpaca

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    df[const.PRED_BINARY] = df[const.PRED].apply(
        lambda x: f(x, args.target_language,
                                           return_string=False))

    return df


def summarize_results(language: str, temperature: float):
    model_prefix = "" if args.model == "gpt35" else f"{args.model}_"

    model_name = args.model

    if args.dataset_name == "healthqa":

        path = osp.join(args.output_dir, "verifiability",
                        f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}_{args.split}_{language}.xlsx")

        # TODO
        # path = osp.join(args.output_dir, "verifiability", f"bak_healthqa_verifiability_dev_English_uses_entailment_prompt.xlsx")

        if not osp.exists(path):
            return None

        df = pd.read_excel(path)

    elif args.dataset_name in ["liveqa", "medicationqa"]:
        df_li = []

        for label in ["positive", "negative"]:
            path = osp.join(args.output_dir, "verifiability",
                            f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}_{language}_{label}.xlsx")

            if not osp.exists(path):
                return None

            if model_name in ["gpt35", "gpt4"]:
                f = map_prediction_to_binary

            elif model_name.startswith("medalpaca"):
                f = map_prediction_to_binary_MedAlpaca

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            df = pd.read_excel(path)
            df[const.LABEL] = const.LABEL2ID[label]



            df[const.PRED] = df[const.PRED].apply(
                lambda x: f(x, language=language))
            # df.to_excel(path, index=False)
            df_li.append(df)

        df = pd.concat(df_li)

    elif args.dataset_name == "mlecqa":

        path = osp.join(args.output_dir, "verifiability",
                        f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}_{args.split}_{language}.xlsx")
        df = pd.read_excel(path)

        new_df = pd.DataFrame()
        new_df[const.PRED] = df[list('ABCDE')].values.reshape(-1)
        new_df[const.ANSWER] = df[const.ANSWER].values.repeat(5)
        new_df[const.OPTION] = list('ABCDE') * len(df)
        new_df[const.LABEL] = new_df[const.ANSWER] == new_df[const.OPTION]

        df = new_df

    # elif args.dataset_name == "medicationqa":
    #     for label in ["positive", "negative"]:
    #         path = osp.join(args.output_dir, "verifiability",
    #                         f"{args.dataset_name}_verifiability_{language}_{label}.xlsx")
    #         df = pd.read_excel(path)
    #
    #         # Map "positive" to 1 and "negative" to 0
    #         df[const.LABEL] = const.LABEL2ID[label]
    #         df[const.PRED] = df[const.PRED].apply(lambda x: map_prediction_to_binary(x, language=language))

    else:
        raise NotImplementedError

    df[const.PRED] = df[const.PRED].apply(
        lambda x: map_prediction_to_binary(x, args.target_language,
                                           return_string=True))
    # df.to_excel(path, index=False)

    # df[const.PRED_BINARY] = df[const.PRED].apply(assign_binary_label_verifiability)
    # TODO
    # df.to_excel(path)
    # Convert string labels to binary values
    df[const.PRED_BINARY] = df[const.PRED].map(ANSWER_MAP).fillna(-1).astype(
        int)

    print(f"Length (original) {len(df)}")
    df = df[df[const.QUESTION].notna()]
    df = df[df[const.PRED].notna()]
    print(f"Length (after drop null questions/predictions) {len(df)}")
    df = df[df[const.PRED_BINARY] != -1]
    print(f"Length (after drop unknown answers) {len(df)}")

    results = {}




    for pos_label in [0, 1]:

        # Get precision, recall, f1, accuracy, and auc scores
        precision, recall, f1, _ = precision_recall_fscore_support(
            df[const.LABEL], df[
                const.PRED_BINARY], average='binary', pos_label=pos_label)

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            df[const.LABEL], df[
                const.PRED_BINARY], average='macro', pos_label=pos_label)
        accuracy = accuracy_score(df[const.LABEL], df[const.PRED_BINARY])
        auc = roc_auc_score(df[const.LABEL], df[const.PRED_BINARY])

        results[pos_label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'auc': auc,
        }

        PLOT_CONFUSION_MATRIX = False

        if PLOT_CONFUSION_MATRIX:

            if pos_label == 1:
                plot_confusion_matrix(df, language=language, pos_label=pos_label,
                                      args=args)

    return results


if __name__ == '__main__':
    project_setup(args)

    # for temperature in const.TEMPERATURES:
    for temperature in [0.0, 1.0]:

        # for language in ["English", "Hindi"]:
        for language in const.LANGUAGES:
            print(f"Language: {language}, Temperature {temperature}")
            args.target_language = language

            model_prefix = "" if args.model == "gpt35" else f"{args.model}_"
            results = summarize_results(args.target_language, temperature)

            if results is None:
                print(f"Not found: {args.dataset_name}\t{args.target_language}\t{temperature}")
                continue

            results = pd.DataFrame(results)

            print(
                f"[{args.dataset_name}]\tLanguage: {args.target_language}, Temperature {temperature}")

            print(results)




            path = osp.join(args.output_dir, "summary",
                            f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}.xlsx")
            if osp.exists(path):
                with pd.ExcelWriter(path, mode='a', engine='openpyxl',
                                    if_sheet_exists='replace') as writer:
                    results.to_excel(writer, sheet_name=language)

            else:
                with pd.ExcelWriter(path, mode='w',
                                    engine='openpyxl') as writer:
                    results.to_excel(writer, sheet_name=language)

            print("Done!")
