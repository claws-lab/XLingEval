import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


import const
from arguments import args
from setup import project_setup
from utils.utils_data import \
    load_data_for_verifiability_heatmap


from matplotlib.colors import LinearSegmentedColormap

colors = [(1, 1, 1), (0.93, 0.0, 0.0)]  # RGB values from white to red
n_bins = [3]  # 3 bins
cmap_name = "custom_diverging"
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

project_setup(args)


def main():

    METRIC_NAMES = ["macro_precision", "macro_recall", "macro_f1",
                    "accuracy", "auc"]

    LANGUAGES = const.LANGUAGES

    model_prefix = f"{args.model}_" if args.model != "gpt35" else ""

    """
    Visualize the verifiability results
    """

    fig, axs = plt.subplots(1, 5, figsize=(28, 5), sharey=True,
                            gridspec_kw={'width_ratios': [5, 5, 5, 5, 5]})

    prefix = "" if args.model == "gpt35" else f"{args.model}_"

    sns.set(font_scale=1.5)

    visual_dir = osp.join(args.output_dir, "summary", "visual", "verifiability")

    os.makedirs(visual_dir, exist_ok=True)

    results_d = load_data_for_verifiability_heatmap(args)

    FONTSIZE = 20
    for metric_name in METRIC_NAMES:
        results_d[metric_name] = pd.DataFrame(columns=LANGUAGES,
                                              index=const.TEMPERATURES).astype(
            float)

    # for temperature in const.TEMPERATURES:
    for temperature in [0.0, 1.0]:

        if args.dataset_name in ["healthqa", "liveqa", "medicationqa"]:
            path = osp.join(args.output_dir, "summary",
                            f"{prefix}{args.dataset_name}_verifiability_temp{temperature}.xlsx")

        else:
            raise NotImplementedError

        for language in LANGUAGES:
            try:
                df = pd.read_excel(path, sheet_name=language, index_col=0)
                for metric_name in METRIC_NAMES:
                    results_d[metric_name].loc[temperature, language] = df.loc[
                        metric_name, 0]

            except:
                pass

    mean_all_metrics = []
    std_all_metrics = []

    for metric_name, results_of_one_metric in \
            results_d.items():



        mean = results_of_one_metric.mean(axis=0)
        std = results_of_one_metric.std(axis=0)

        mean_all_metrics.append(mean)
        std_all_metrics.append(std)

    mean = pd.concat(mean_all_metrics, axis=1)
    std = pd.concat(std_all_metrics, axis=1)

    mean.columns = const.VERIFIABILITY_METRICS_VISUALIZATION
    std.columns = const.VERIFIABILITY_METRICS_VISUALIZATION

    df_merged = pd.DataFrame()

    for col in mean.columns:
        mean[col] = mean[col].astype(np.float64).map(
            "{:.4f}".format)
        std[col] = std[col].astype(np.float64).map(
            "{:.4f}".format)

        df_merged[col] = mean[col].astype(str) + ' + ' + std[
            col].astype(str)


    print(df_merged)

    path = f"outputs/summary/{model_prefix}verifiability_aggregated.xlsx"

    if osp.exists(path):
        with pd.ExcelWriter(path, mode='a', if_sheet_exists="replace") as writer:
            df_merged.to_excel(writer, sheet_name=args.dataset_name)

    else:
        with pd.ExcelWriter(path, mode='w') as writer:
            df_merged.to_excel(writer, sheet_name=args.dataset_name)


    if not DO_PLOT:
        return



    for i, metric_name in enumerate(const.VERIFIABILITY_METRICS_VISUALIZATION):
        results_d[metric_name].index = results_d[metric_name].index.astype(str)
        results_d[metric_name].columns = const.LANGUAGE_CODES

        yticks = True if i == 0 else False

        sns.heatmap(results_d[metric_name].astype(float), annot=True, fmt=".2f",
                    cmap=cmap, cbar=False, ax=axs[i], alpha=0.5,
                    vmax=0.95, vmin=0.6, xticklabels=True, yticklabels=yticks, annot_kws={"color": "black"}
                    )


        axs[i].set_xticklabels(axs[i].get_xticklabels(), fontsize=FONTSIZE)


        axs[i].set_title(
            f"{const.METRIC_NAME2FULLNAME[metric_name]}",
            fontsize=FONTSIZE)

        if i == 0:
            axs[i].set_yticklabels(axs[i].get_yticklabels(),
                                   fontsize=FONTSIZE)

    # Remove x and y ticks
    # for ax in axs:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    # Add shared x and y labels
    # fig.text(0.5, 0.01, "Language", ha="center", fontsize=20)
    fig.text(0.09, 0.5, "Temperature " + r"$\tau$", va="center",
             rotation="vertical", fontsize=FONTSIZE)

    fig.subplots_adjust(right=0.88)

    # Create colorbar
    cbar_ax = fig.add_axes([0.89, 0.1, 0.015, 0.8])
    fig.colorbar(axs[0].collections[0], cax=cbar_ax)
    # plt.tight_layout()

    # plt.show()
    plt.savefig(
        osp.join(visual_dir, f"{prefix}{args.dataset_name}_verifiability.pdf"),
        dpi=300)


if __name__ == "__main__":
    DO_PLOT = False
    for dataset_name in ["liveqa"]:
    # for dataset_name in ["healthqa", "liveqa", "medicationqa"]:
        args.dataset_name = dataset_name
        main()