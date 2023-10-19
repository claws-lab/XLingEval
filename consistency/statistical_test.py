import itertools
import os
import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from dataloader.load_data import load_consistency_results

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import const
from arguments import args
from setup import project_setup
import warnings

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

for num_topics in [10, 20, 50, 100, 200, 500]:
    const.METRIC_NAME2FULLNAME[
        f'lda{num_topics}_mean'] = f"Mean Topical Similarity (LDA with {num_topics} topics)"
    const.METRIC_NAME2FULLNAME[
        f'lda{num_topics}_std'] = f"Std. of Topical Similarity (LDA with {num_topics} topics)"

# METRIC_NAMES = ['bertscore_P']

project_setup(args)


def main(temperature: float, metric_name: str):
    # This dataframe might contain NaN values
    original_df = pd.concat(
        [temp2lang_and_df[temperature][language][metric_name] for language in
         const.LANGUAGES],
        axis=1)

    # Drop any columns that contain any np.NaN values
    df = original_df.dropna(axis=0, how="any")

    print(f"#Rows before/after dropna: {len(original_df)}/{len(df)}")

    df.columns = const.LANGUAGES

    # NOTE: we need to transpose the original dataframe here!
    df = pd.DataFrame([df.values.T.flatten(), df.index.tolist() * 4,
                       np.repeat(const.LANGUAGES, len(df))],
                      index=[metric_name, const.ID, const.LANGUAGE]).T.astype({
        metric_name: float
    })

    plt.figure(figsize=(6, 6))

    # Set the style and color palette
    sns.set(style="whitegrid", font_scale=1.5)
    sns.set_palette('Set3')

    # Create larger figure

    # Create boxplot

    if metric_name in ["bert_sim", "bertscore_F1", "unigram_jaccard", "bigram_jaccard", "lda20_mean", "hdp_mean"]:
        boxplot = sns.boxplot(x=const.LANGUAGE, y=metric_name, data=df,
                              showfliers=True,  # Whether to show the outliers
                              fliersize=4,
                              width=0.8)



    elif metric_name in []:

        violinplot = sns.violinplot(x=const.LANGUAGE, y=metric_name, data=df, palette="Set3")

    elif metric_name in []:
        swarmplot = sns.swarmplot(x=const.LANGUAGE, y=metric_name, data=df, palette="Set3")

        # Loop through the artists (dots) in the swarmplot and modify the edge width
        for artist in swarmplot.collections:
            artist.set_linewidth(0.5)

    # Add stripplot (i.e., a scatterplot where one variable is categorical) for individual scores
    # sns.stripplot(x=const.LANGUAGE, y=metric_name, data=df,
    #               size=1, jitter=True, edgecolor="gray", linewidth=0.1)

    # Set the title and labels
    plt.title(r"$\tau$" + f" = {temperature}", fontsize=20)
    plt.xlabel(const.LANGUAGE, fontsize=15)
    plt.ylabel(const.METRIC_NAME2FULLNAME[metric_name], fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if metric_name in ["bert_sim"]:
        plt.ylim(0.75, 1.02)


    elif metric_name in ["bertscore_F1"]:
        plt.ylim(0.2, 1.02)

    elif metric_name in ["unigram_jaccard", "bigram_jaccard", "lda20_mean", "hdp_mean"]:
        plt.ylim(0., 1.02)

    else:
        plt.ylim(0., 1.02)

    plt.tight_layout()

    plt.savefig(
        osp.join(path_results,
                 f"{args.dataset_name}_temp{temperature}_{metric_name}_plot.pdf"),
        dpi=300)
    # plt.show()

    # Run the ANOVA

    anova_table = tukey_df = unpaired_t_test_df = None


    """
    
    - `F-value`: it decides whether the hypothesis of all population means being equal is rejected or not. The larger the F-value, the more likely it is that the differences among group means are due to something other than chance. 
    
    - `PR(>F)`: The p-value associated with the F-statistic. It's used to test the null hypothesis that all of the population means are equal. A small p-value (typically ≤ 0.05) indicates strong evidence against the null hypothesis, so you reject the null hypothesis. A large p-value (> 0.05) indicates weak evidence against the null hypothesis, so you fail to reject the null hypothesis. 
    """

    if DO_ANOVA:
        model = ols(f'{metric_name} ~ C({const.LANGUAGE})',
                    data=df[[metric_name, const.LANGUAGE]]).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(anova_table)

    """
    To interpret our results in Tukey HSD:
    
    - `meandiff`: the difference in means between the two groups. A positive value indicates that the mean of 'group2' is larger than 'group1' (group2 - group1 > 0), and a negative value means the opposite.
    - `reject`: True if the null hypothesis is rejected (i.e., the p-value is less than your chosen alpha level), and False otherwise.`
    - 'p-adj' is the adjusted p-value for that comparison, after adjusting for multiple comparisons (which is what the Tukey HSD test does).
    """

    if DO_PAIRED_TEST:
        tukey_result = pairwise_tukeyhsd(df[metric_name].tolist(),
                                         df[const.LANGUAGE].tolist(), alpha=0.05)
        print(tukey_result)

        # Pairwise comparison plot
        # Convert Tukey HSD result to a DataFrame
        tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                columns=tukey_result._results_table.data[0])

        tukey_df = tukey_df.rename({
            "group1": "language1",
            "group2": "language2",
            "meandiff": "MD",
        })

        tukey_df["95% CI"] = tukey_df.apply(lambda x: f"({x['lower']:.4f}, {x['upper']:.4f})", axis=1)

    if DO_UNPAIRED_TEST:

        unpaired_t_test_df = pd.DataFrame(
            columns=["language1", "language2", 't-statistic', 'p-value'])

        for lang1, lang2 in itertools.combinations(const.LANGUAGES, 2):
            group1 = df[df[const.LANGUAGE] == lang1][metric_name]
            group2 = df[df[const.LANGUAGE] == lang2][metric_name]

            """
            `nan_policy`: if set to 'omit', we exclude any missing values from the data.
            """

            t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')

            print(f"t-test between {lang1} and {lang2}:")
            print(f"\tt statistic: {t_stat}")
            print(f"\tp-value: {p_val}")

            unpaired_t_test_df = unpaired_t_test_df.append({
                "language1": lang1,
                "language2": lang2,
                "t-statistic": t_stat,
                "p-value": p_val
            }, ignore_index=True)
            unpaired_t_test_df["p-value"] = unpaired_t_test_df["p-value"].astype(np.float64).map("{:.2e}".format)

    return anova_table, tukey_df, unpaired_t_test_df


if __name__ == "__main__":

    temp2lang_and_df = {}
    temp2lang_and_df = load_consistency_results(temp2lang_and_df)

    tukey_df_d = {}
    unpaired_t_test_df_d = {}

    anova_d = {
        temperature: pd.DataFrame(columns=["metric", "F-value", "p-value"]) for
        temperature in const.TEMPERATURES}

    path_results = osp.join(args.output_dir, "summary", args.dataset_name)

    DO_ANOVA = True
    DO_UNPAIRED_TEST = True
    DO_PAIRED_TEST = True

    os.makedirs(path_results, exist_ok=True)

    for metric_name in ["bert_sim", "bertscore_F1", "unigram_jaccard", "bigram_jaccard", "length_mean", "lda20_mean",
                        "hdp_mean"]:

        for temperature in const.TEMPERATURES:
            print("-" * 20)
            print(f"> Metric {metric_name}, Temperature {temperature}")
            print("-" * 20)

            anova_table, tukey_df, unpaired_t_test_df = main(temperature,
                                                             metric_name)

            p_anova = anova_table['PR(>F)'][f'C({const.LANGUAGE})']
            F_value = anova_table['F'][f'C({const.LANGUAGE})']

            tukey_df_d[(temperature, metric_name)] = tukey_df.copy()

            unpaired_t_test_df_d[
                (temperature, metric_name)] = unpaired_t_test_df.copy()

            anova_d[temperature] = anova_d[temperature].append({
                "metric": metric_name,
                "F-value": F_value,
                "p-value": p_anova
            }, ignore_index=True)

        print("\n" + "=" * 20 + "\n")

    METRICS = ["bert_sim", "bertscore_F1", "unigram_jaccard",
               "bigram_jaccard", "length_mean", "lda20_mean", "hdp_mean"]

    for temperature in const.TEMPERATURES:
        anova_d[temperature] = anova_d[temperature].set_index("metric").T[METRICS]
        anova_d[temperature]['temperature'] = temperature

    anova_df = pd.concat(anova_d.values())

    for metric_name in METRICS:
        anova_df.loc[anova_df.index == "F-value", metric_name] = anova_df.loc[
            anova_df.index == "F-value", metric_name].map(
            "{:.2f}".format)

        anova_df.loc[anova_df.index == "p-value", metric_name] = \
            anova_df.loc[anova_df.index == "p-value", metric_name].map(
                "{:.2e}".format)

    path_anova = osp.join(path_results, f"{args.dataset_name}_anova.xlsx")

    if osp.exists(path_anova):
        with pd.ExcelWriter(path_anova, engine='openpyxl',
                            mode='a',
                            if_sheet_exists='replace',
                            ) as writer:
            for temperature in const.TEMPERATURES:
                anova_d[temperature] = anova_d[temperature].T

                anova_d[temperature].to_excel(writer,
                                              sheet_name=f"temp{temperature}")

            anova_df.to_excel(writer, sheet_name="summary")

    else:
        with pd.ExcelWriter(path_anova, engine='openpyxl',
                            mode='w',
                            ) as writer:
            for temperature in const.TEMPERATURES:
                anova_d[temperature] = anova_d[temperature].set_index("metric").T

                anova_d[temperature].to_excel(writer,
                                              sheet_name=f"temp{temperature}")

            anova_df.to_excel(writer, sheet_name="summary")

    for temperature in const.TEMPERATURES:
        path_tukey = osp.join(path_results,
                              f"{args.dataset_name}_temp{temperature}_tukey.xlsx")
        path_unpaired = osp.join(path_results,
                                 f"{args.dataset_name}_temp{temperature}_unpaired.xlsx")

        with pd.ExcelWriter(path_tukey, engine='openpyxl',
                            mode='w') as writer1, pd.ExcelWriter(path_unpaired,
                                                                 engine='openpyxl',
                                                                 mode='w') as writer2:

            # for metric_name in const.METRIC_NAME2FULLNAME:
            for metric_name in METRICS:

                if DO_PAIRED_TEST:
                    tukey_df_d[(temperature, metric_name)].to_excel(writer1,
                                                                    index=False,
                                                                    sheet_name=f"{metric_name}")

                if DO_UNPAIRED_TEST:
                    unpaired_t_test_df_d[(temperature, metric_name)].to_excel(
                        writer2,
                        index=False,
                        sheet_name=f"{metric_name}")

    # Export results of ANOVA for plotting

    df_all_temperatures_and_metrics = []

    path_summary_tukey = osp.join(path_results,
                                  f"paper_{args.dataset_name}_tukey.xlsx")

    with pd.ExcelWriter(path_summary_tukey, engine='openpyxl',
                        mode='w') as writer:

        for temperature in const.TEMPERATURES:

            df = tukey_df_d[(0.0, "bert_sim")][["group1", "group2"]].copy()

            for metric_name in ["bert_sim", "bertscore_F1", "unigram_jaccard",
                                "bigram_jaccard", "lda20_mean",
                                "hdp_mean"]:

                for field in ["95% CI", 'meandiff', "p-adj", "reject"]:
                    df = pd.concat([df, tukey_df_d[(temperature, metric_name)][field]], axis=1)

                df.to_excel(writer, index=False,
                            sheet_name=f"temp{temperature}")

    # Export results of HSD Tukey's test for plotting

    with pd.ExcelWriter(path_summary_tukey, engine='openpyxl',
                        mode='w') as writer:
        for temperature in const.TEMPERATURES:

            df_tukey_all = pd.concat([tukey_df_d[(temperature, metric_name)]['p-adj'] for metric_name in
                                      ["bert_sim", "bertscore_F1", "unigram_jaccard",
                                       "bigram_jaccard", "lda20_mean",
                                       "hdp_mean"]], axis=1)

            for col in ['group1', 'group2', 'p-adj']:
                df_tukey_all[col] = tukey_df_d[(temperature, METRICS[0])][col]

            df_tukey_all['temperature'] = temperature

            df_tukey_all.to_excel(writer, index=False,
                                  sheet_name=f"temp{temperature}")

    path_summary_unpaired = osp.join(path_results,
                                     f"paper_{args.dataset_name}_temp{temperature}_unpaired.xlsx")
    with pd.ExcelWriter(path_summary_unpaired, engine='openpyxl', mode='w') as writer:
        for temperature in const.TEMPERATURES:
            df_unpaired_all = unpaired_t_test_df_d[(temperature, metric_name)][['language1', 'language2']].copy()

            df_unpaired_all = pd.concat(
                [df_unpaired_all] + [unpaired_t_test_df_d[(temperature, metric_name)][["t-statistic", "p-value"]] for
                                     metric_name in METRICS], axis=1)

            df_unpaired_all.to_excel(writer, sheet_name=f"temp{temperature}", index=False, )

    df_all_temperatures_and_metrics = pd.concat(df_all_temperatures_and_metrics)

    print("Done!")
