import os.path as osp
import string

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, \
    roc_auc_score, confusion_matrix

import const
from arguments import args
from setup import project_setup
from utils.utils_misc import map_prediction_to_binary_MedAlpaca, get_model_prefix


def map_prediction_to_binary(x, return_string=True, return_unknown=False):
    """Map the predictions to binary (Yes/No/Unknown).

    GPT models tend to generate more verbose contents than we instructed them to do.
    This function maps their verbose answers to categorical outputs.

    Args:
        x (str): The prediction string.
        return_string (bool): Whether to return the result as a string.
        return_unknown (bool): Whether to return 'Unknown' when result is ambiguous.

    Returns:
        str or int: Mapped prediction in either string or integer format.
    """

    ANSWER_MAP = {
        "Yes": {"Yes"'Entailment', 'Sí', 'Yes', 'जी हाँ', 'निश्चित रूप से हाँ।', 'हाँ', '可能性很大', '可能是正确的',
                '对',
                '是的', '正确'
                },
        "No": {'No', 'Not entailment', "I'm sorry, but i am unable to understand",
               "I'm sorry, but the response you provided is not related to the question",
               "I'm sorry, but your response does not seem to be related to the question asked", 'नहीं', 'नो',
               'माफ़ कीजिए, लेकिन यह प्रतिक्रिया दिए गए प्रश्न से मेल नहीं खाती है',
               'यह उत्तर दिया गया प्रश्न संबंधित नहीं है। कृपया सही प्रश्न पूछें',
               'यह उत्तर दिया गया प्रश्न से मेल नहीं खाता है', 'यह उत्तर नहीं है',
               'यह उत्तर प्रश्न के साथ संबंधित नहीं है। कृपया पुनः प्रश्न देखें', '不是', '不正确', '以上都不', '错误'},

        "Unknown": {'Unknown', 'Depends', 'N/a', 'कारण पता नहीं है', '不确定', '无法判断'}
    }

    if not isinstance(x, str):
        return -1 if not return_string else "Unknown" if return_unknown else x

    x_lower = x.lower()

    if any(prefix in x_lower for prefix in map(str.lower, ANSWER_MAP["No"])) and "yes" not in x_lower:
        return "No" if return_string else 0

    if any(prefix in x_lower for prefix in map(str.lower, ANSWER_MAP["Yes"])) and "no" not in x_lower:
        return "Yes" if return_string else 1

    if any(prefix in x_lower for prefix in map(str.lower, ANSWER_MAP["Unknown"])):
        return "Unknown" if return_string else -1

    return -1 if not return_string else ("Unknown" if return_unknown else x)


def map_prediction_to_binary_med_alpaca(x, return_string=True, return_unknown=False):
    """Map the predictions of MedAlpaca to binary (Yes/No/Unknown).

    MedAlpaca's answers are significantly different from GPT-3.5's,
    requiring a different mapping format.

    Args:
        x (str): The prediction string.
        return_string (bool): Whether to return the result as a string.
        return_unknown (bool): Whether to return 'Unknown' when result is ambiguous.

    Returns:
        str or int: Mapped prediction in either string or integer format.
    """

    ANSWER_MAP = {
        'Yes': {'Yes', 'Entailment'},
        'No': {'No', 'Not entailment'},
        'Unknown': {'Unknown', 'Depends', 'N/a'}
    }

    if isinstance(x, str):
        x = x.split('### Input')[-1].split('Question')[-1].split('Response')[-1]

    if not isinstance(x, str):
        return -1 if not return_string else 'Unknown' if return_unknown else x

    if all(any(segment in x for segment in ANSWER_MAP[key]) for key in ['Yes', 'No']):
        return 'Unknown' if return_string else -1

    for key, value in ANSWER_MAP.items():
        if any(segment in x for segment in value):
            return key if return_string else 1 if key == 'Yes' else 0 if key == 'No' else -1

    return -1 if not return_string else ('Unknown' if return_unknown else x)


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
    plt.show()


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


def summarize_verifiability_results(language: str, temperature: float):
    model_prefix = "" if args.model == "gpt35" else f"{args.model}_"

    model_name = args.model

    if args.dataset_name == "healthqa":

        path = osp.join(args.output_dir, "verifiability",
                        f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}_{args.split}_{language}.xlsx")

    elif args.dataset_name in ["liveqa", "medicationqa"]:

        path = osp.join(args.output_dir, "verifiability",
                        f"{model_prefix}{args.dataset_name}_verifiability_temp{temperature}.xlsx")


    else:
        raise NotImplementedError

    if not osp.exists(path):
        return None

    df = pd.read_excel(path, sheet_name=language)

    if model_name in ["gpt35", "gpt4"]:
        f = map_prediction_to_binary

    elif model_name.startswith("medalpaca"):
        f = map_prediction_to_binary_MedAlpaca

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    df[const.PRED_BINARY] = df[const.PRED].apply(
        lambda x: f(x, return_string=False))

    df[const.PRED_BINARY] = df[const.PRED_BINARY].fillna(-1).astype(
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

        if PLOT_CONFUSION_MATRIX and pos_label == 1:
            plot_confusion_matrix(df, language=language, pos_label=pos_label,
                                  args=args)

    return results


if __name__ == '__main__':
    project_setup()
    PLOT_CONFUSION_MATRIX = True

    for temperature in [0.0, 0.25, 0.5, 0.75, 1.0]:

        for language in ["English", "Spanish", "Chinese", "Hindi"]:
            print(f"Language: {language}, Temperature {temperature}")
            args.target_language = language

            model_prefix = "" if args.model == "gpt35" else f"{args.model}_"
            results = summarize_verifiability_results(args.target_language, temperature)

            if results is None:
                print(f"Not found: {args.dataset_name}\t{args.target_language}\t{temperature}")
                continue

            results = pd.DataFrame(results)

            print(
                f"[{args.dataset_name}]\tLanguage: {args.target_language}, Temperature {temperature}")

            print(results)

            path = osp.join(args.output_dir, "summary",
                            f"{get_model_prefix(args)}{args.dataset_name}_verifiability_temp{temperature}.xlsx")
            if osp.exists(path):
                with pd.ExcelWriter(path, mode='a', engine='openpyxl',
                                    if_sheet_exists='replace') as writer:
                    results.to_excel(writer, sheet_name=language)

            else:
                with pd.ExcelWriter(path, mode='w',
                                    engine='openpyxl') as writer:
                    results.to_excel(writer, sheet_name=language)

            print("Done!")
