import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataloader.load_data import load_consistency_results

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import const
from arguments import args
from setup import project_setup
import warnings

os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"

warnings.filterwarnings("ignore",
                        message="Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertModel*")

METRIC_NAMES = [
    'bert_sim', 'bertscore_P', 'bertscore_R', 'bertscore_F1', 'unigram_jaccard',
    'bigram_jaccard', 'length_mean', 'length_std', 'hdp_mean', 'hdp_std',
    'lda10_mean', 'lda10_std', 'lda20_mean', 'lda20_std', 'lda50_mean',
    'lda50_std',
    'lda100_mean', 'lda100_std', 'lda200_mean', 'lda200_std', 'lda500_mean',
    'lda500_std'
]

FONTSIZE = 24


def main():
    # Important: this is needed for LaTeX rendering
    plt.rcParams["text.usetex"] = True

    path_results = osp.join(args.output_dir, "summary", "visual", "consistency")
    temp2lang_and_df = {}
    temp2lang_and_df = load_consistency_results(temp2lang_and_df)

    # Set the style and color palette
    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_palette('Set3')


    for i, metric_name in enumerate(
            ["bert_sim", "bertscore_F1", "unigram_jaccard", "bigram_jaccard",
             "lda20_mean", "hdp_mean"]):

        fig, axs = plt.subplots(1, 5, figsize=(28, 5), sharey=True,
                                gridspec_kw={'width_ratios': [5, 5, 5, 5, 5]})

        for j, temperature in enumerate(const.TEMPERATURES):

            # This dataframe might contain NaN values
            original_df = pd.concat(
                [temp2lang_and_df[temperature][language][metric_name] for
                 language in
                 const.LANGUAGES],
                axis=1)

            # Drop any columns that contain any np.NaN values
            df = original_df.dropna(axis=0, how="any")

            print(f"#Rows before/after dropna: {len(original_df)}/{len(df)}")

            df.columns = const.LANGUAGE_CODES

            # NOTE: we need to transpose the original dataframe here!
            df = pd.DataFrame([df.values.T.flatten(), df.index.tolist() * 4,
                               np.repeat(const.LANGUAGE_CODES, len(df))],
                              index=[metric_name, const.ID,
                                     const.LANGUAGE]).T.astype({
                metric_name: float
            })

            sns.boxplot(x=const.LANGUAGE, y=metric_name, data=df, ax=axs[j],
                        showfliers=True,  # Whether to show the outliers
                        fliersize=4,
                        width=0.8)

            axs[j].set_xlabel('')
            axs[j].set_ylabel('')

            axs[j].set_xticklabels(axs[j].get_xticklabels(), fontsize=FONTSIZE)

            # Add stripplot (i.e., a scatterplot where one variable is categorical) for individual scores
            # sns.stripplot(x=const.LANGUAGE, y=metric_name, data=df,
            #               size=1, jitter=True, edgecolor="gray", linewidth=0.1)

            # Set the title and labels

            axs[j].set_title(
                r"$\tau$" + f" = {temperature}",
                fontsize=FONTSIZE)

            # Only annotate the first boxplot
            if j == 0:
                fig.text(0.087, 0.5, const.METRIC_NAME2FULLNAME[metric_name],
                         va="center",
                         rotation="vertical", fontsize=FONTSIZE)

                axs[j].set_yticklabels(axs[j].get_yticklabels(),
                                       fontsize=FONTSIZE)

                # plt.ylabel(const.METRIC_NAME2FULLNAME[metric_name], fontsize=16)

            if metric_name in ["bert_sim"]:
                plt.ylim(0.75, 1.01)


            elif metric_name in ["bertscore_F1"]:
                plt.ylim(0.2, 1.01)

            elif metric_name in ["unigram_jaccard", "bigram_jaccard",
                                 "lda20_mean", "hdp_mean"]:
                plt.ylim(0., 1.01)

            else:
                plt.ylim(0., 1.01)

        plt.savefig(
            osp.join(path_results,
                     f"boxplot_{args.dataset_name}_{metric_name}.pdf"),
            dpi=300)

        del fig, axs


if __name__ == "__main__":
    project_setup(args)

    main()
