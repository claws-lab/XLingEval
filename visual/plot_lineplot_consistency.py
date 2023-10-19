import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import const
from arguments import args
from setup import project_setup
from utils.utils_misc import capitalize_dataset_name

project_setup(args)

if __name__ == "__main__":
    visual_dir = osp.join(args.output_dir, "summary", "visual", "consistency")

    os.makedirs(visual_dir, exist_ok=True)

    results_d = {}

    METRIC_NAMES = const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING

    for metric_name in const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING:
        results_d[metric_name] = pd.DataFrame(columns=const.LANGUAGES,
                                              index=const.TEMPERATURES).astype(
            float)

        results_d[metric_name].index.rename("temperature", inplace=True)

    for temperature in const.TEMPERATURES:

        if args.dataset_name in ["liveqa", "medicationqa"]:
            path = osp.join(args.output_dir, "summary",
                            f"{args.dataset_name}_consistency_temp{temperature}.xlsx")

        elif args.dataset_name in ["healthqa"]:
            path = osp.join(args.output_dir, "summary",
                            f"{args.dataset_name}_consistency_{args.split}_temp{temperature}.xlsx")

        else:
            raise NotImplementedError

        for language in const.LANGUAGES:
            df = pd.read_excel(path, sheet_name=language, index_col=0)

            for metric_name in const.CONSISTENCY_METRICS_SIMILARITY + const.CONSISTENCY_METRICS_TOPIC_MODELING:
                average_metrics = df.mean(axis=0)

                results_d[metric_name].loc[temperature, language] = \
                average_metrics.loc[metric_name]

    for metric_name in METRIC_NAMES:
        plt.figure(figsize=(6, 10))



        sns.set(font_scale=2.0)
        results_d[metric_name].columns.name = const.LANGUAGE

        sns.lineplot(# x=const.LANGUAGE, y=metric_name,
                     data=results_d[metric_name], markers=True, marker='o', markersize=12, linewidth=2)

        plt.title(
            f"{const.METRIC_NAME2FULLNAME[metric_name]}",
            fontsize=20)

        plt.xlabel(const.TEMPERATURE, fontsize=20)


        # Set the width of y-ticks to 0.25
        plt.xticks(const.TEMPERATURES)
        plt.tight_layout()


        plt.savefig(
            osp.join(visual_dir, f"co_{args.dataset_name}_{metric_name}.pdf"),
            dpi=300)

        plt.close()

