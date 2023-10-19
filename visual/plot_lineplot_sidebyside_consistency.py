import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

import numpy as np
import const
from arguments import args
from setup import project_setup

sns.set_theme()
project_setup(args)


def main():

    # Important: this is needed for LaTeX rendering
    plt.rcParams["text.usetex"] = True

    FONTSIZE=24
    visual_dir = osp.join(args.output_dir, "summary", "visual", "consistency")

    os.makedirs(visual_dir, exist_ok=True)

    results_d = {}

    METRIC_NAMES = ['bigram_jaccard', 'length_mean', 'bertscore_F1', 'bert_sim',
                    'hdp_mean', 'lda20_mean', ]

    model_prefix = f"{args.model}_" if args.model != "gpt35" else ""

    path_cache = osp.join(visual_dir, f"{model_prefix}consistency_results_d.pt")


    if args.model == "gpt35":
        DATASETS = ["liveqa", "medicationqa", "healthqa"]
        TEMPERATURES = [0.0, 0.25, 0.5, 0.75, 1.0]

    else:
        DATASETS = ["liveqa"]
        TEMPERATURES = [0.0, 1.0]

    if osp.exists(path_cache):
        results_d = torch.load(path_cache)

    else:

        for dataset_name in DATASETS:

            results_d[dataset_name] = {}

            for metric_name in const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING:
                results_d[dataset_name][metric_name] = pd.DataFrame(
                    columns=const.LANGUAGES,
                    index=const.TEMPERATURES).astype(
                    float)

                results_d[dataset_name][metric_name].index.rename("temperature",
                                                                  inplace=True)

        for dataset_name in DATASETS:

            for temperature in TEMPERATURES:

                if dataset_name in ["liveqa", "medicationqa"]:
                    path = osp.join(args.output_dir, "summary",
                                    f"{model_prefix}{dataset_name}_consistency_temp{temperature}.xlsx")

                elif dataset_name in ["healthqa"]:
                    path = osp.join(args.output_dir, "summary",
                                    f"{model_prefix}{dataset_name}_consistency_{args.split}_temp{temperature}.xlsx")

                else:
                    raise NotImplementedError

                for language in const.LANGUAGES:
                    df = pd.read_excel(path, sheet_name=language, index_col=0)

                    for metric_name in const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING:
                        average_metrics = df.mean(axis=0)

                        results_d[dataset_name][metric_name].loc[
                            temperature, language] = \
                            average_metrics.loc[metric_name]

        torch.save(results_d, path_cache)

    # Aggregate the consistency results over different temperatures.

    with pd.ExcelWriter(osp.join(f"outputs/summary/{model_prefix}consistency_aggregated.xlsx")) as writer:

        for dataset_name, results_of_one_dataset in results_d.items():
            aggregated_one_dataset, decrease_one_dataset = [], []

            for metric_name, _ in results_of_one_dataset.items():
                aggregated = results_of_one_dataset[metric_name].mean(axis=0)
                decrease = (aggregated - aggregated.loc["English"]) / aggregated.loc["English"]

                aggregated_one_dataset += [aggregated]
                decrease_one_dataset += [decrease]

            aggregated = pd.concat(aggregated_one_dataset, axis=1)
            decrease = pd.concat(decrease_one_dataset, axis=1)

            aggregated.columns = list(results_of_one_dataset.keys())
            decrease.columns = list(results_of_one_dataset.keys())

            columns = ['bert_sim', 'bertscore_F1','unigram_jaccard', 'bigram_jaccard', 'length_mean', 'hdp_mean', 'lda10_mean', 'lda20_mean', 'lda50_mean', 'lda100_mean','lda200_mean', 'lda500_mean',]


            aggregated = aggregated[columns]
            decrease = decrease[columns]

            for col in columns:
                aggregated[col] = aggregated[col].astype(np.float64).map("{:.4f}".format)
                decrease[col] = (decrease[col] * 100).astype(np.float64).map("{:.1f}".format)


            print("=" * 30)
            print(f"Dataset: {dataset_name}")
            print("Aggregated")
            print(aggregated)
            print("Decrease")
            print(decrease)

            df_merged = pd.DataFrame()

            for col in aggregated.columns:
                df_merged[col] = aggregated[col].astype(str) + '/' + decrease[
                    col].astype(str) + '%'

            print(df_merged)
            df_merged.to_excel(writer, sheet_name=f"{dataset_name}")


    if not DO_PLOT:
        return


    for dataset_name in const.DATASET2LENGTH:
        fig, axs = plt.subplots(1, 6, figsize=(32, 6),
                                gridspec_kw={
                                    'width_ratios': [6, 6, 6, 6, 6, 6]})

        lines = []  # Collect the lines for the joint legend

        for i, metric_name in enumerate(METRIC_NAMES):
            sns.set(font_scale=2.0)
            results_d[dataset_name][metric_name].columns.name = const.LANGUAGE

            lineplot = sns.lineplot(data=results_d[dataset_name][metric_name],
                                    ax=axs[i], markers=True, marker='o',
                                    markersize=16, linewidth=2, legend=False)

            lines.extend(
                lineplot.get_lines())  # Collect lines for the joint legend

            axs[i].set_title(
                f"{const.METRIC_NAME2FULLNAME[metric_name]}",
                fontsize=FONTSIZE)

            axs[i].set_xlabel("Temperature " + r"$\tau$", fontsize=FONTSIZE)


            # Set the width of y-ticks to 0.25
            axs[i].set_xticks(const.TEMPERATURES)
            axs[i].tick_params(axis='x',
                               labelsize=FONTSIZE)  # Adjust x-tick font size
            axs[i].tick_params(axis='y',
                               labelsize=FONTSIZE)  # Adjust y-tick font size

        fig.subplots_adjust(top=0.9, hspace=0.1)
        joint_legend = fig.legend(
            handles=lines,
            labels=[f"{lang} ({code})" for lang, code in zip(const.LANGUAGES, const.LANGUAGE_CODES)],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(const.LANGUAGES),
            fontsize=FONTSIZE,
        )
        joint_legend.get_frame().set_linewidth(0.0)

        # joint_legend.get_frame().set_facecolor('white') # white background
        joint_legend.get_frame().set_alpha(0.0)  # transparent background

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        plt.savefig(
            osp.join(visual_dir, f"consistency_{dataset_name}.pdf"),
            dpi=300)

        plt.close()


if __name__ == "__main__":
    DO_PLOT = True
    main()