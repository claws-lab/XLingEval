import os
import os.path as osp

import matplotlib.pyplot as plt
import seaborn as sns

import const
from arguments import args
from setup import project_setup
from utils.utils_misc import capitalize_dataset_name
from utils.utils_data import \
    load_data_for_verifiability_heatmap

project_setup(args)



if __name__ == "__main__":
    """
    Visualize the verifiability results
    """

    visual_dir = osp.join(args.output_dir, "summary", "visual", "verifiability")

    os.makedirs(visual_dir, exist_ok=True)

    results_d = load_data_for_verifiability_heatmap(args)


    for metric_name in const.VERIFIABILITY_METRICS_VISUALIZATION:
        plt.figure(figsize=(8, 6))

        sns.set(font_scale=1.5)

        sns.heatmap(results_d[metric_name].astype(float), annot=True, fmt=".2f",
                    cmap="rocket", cbar=True,
                    vmax=0.95, vmin=0.6
                    )

        plt.xlabel("Language", fontsize=20)

        plt.ylabel("Temperature " + r"$\tau$", fontsize=20)
        plt.title(
            f"{capitalize_dataset_name(args.dataset_name)} {const.METRIC_NAME2FULLNAME[metric_name]}",
            fontsize=20)

        plt.tight_layout()

        # plt.show()
        plt.savefig(
            osp.join(visual_dir, f"{args.dataset_name}_{metric_name}.pdf"),
            dpi=300)

    print("Done!")
