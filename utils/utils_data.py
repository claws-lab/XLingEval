import os.path as osp

import pandas as pd

import const


def load_data_for_verifiability_heatmap(args):
    results_d = {}

    prefix = "" if args.model == "gpt35" else f"{args.model}_"

    for metric_name in const.VERIFIABILITY_METRICS_VISUALIZATION:
        results_d[metric_name] = pd.DataFrame(columns=const.LANGUAGES,
                                              index=const.TEMPERATURES).astype(
            float)

    for temperature in const.TEMPERATURES:

        if args.dataset_name in ["healthqa", "liveqa", "medicationqa"]:
            path = osp.join(args.output_dir, "summary",
                            f"{prefix}{args.dataset_name}_verifiability_temp{temperature}.xlsx")

        else:
            raise NotImplementedError

        for language in const.LANGUAGES:
            try:

                df = pd.read_excel(path, sheet_name=language, index_col=0)

            except:
                print(
                    f"Not found: [{args.dataset_name}] temp {temperature}, language {language}")
                continue

            for metric_name in const.VERIFIABILITY_METRICS_VISUALIZATION:
                results_d[metric_name].loc[temperature, language] = df.loc[
                    metric_name, 0]

    return results_d
