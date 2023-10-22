from utils.utils_misc import check_cwd


def project_setup():
    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)

    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 20)


KEYS = [
    {
        "name": "project-name",
        "deployment_id": "deployment 1",
        "key1": "enter your key 1 here",
        "key2": "enter your key 2 here",
        "region": "East US",
    },

    {
        "name": "project-name",
        "deployment_id": "deployment 2",
        "key1": "enter your key 1 here",
        "key2": "enter your key 2 here",
        "region": "East US",
    },

    {
        "name": "project-name",
        "deployment_id": "deployment 3",
        "key1": "enter your key 1 here",
        "key2": "enter your key 2 here",
        "region": "East US",
    },

    {
        "name": "project-name",
        "deployment_id": "deployment 4",
        "key1": "enter your key 1 here",
        "key2": "enter your key 2 here",
        "region": "East US",
    }
]

def openai_setup():

    import openai


    openai_objs = []

    for key in KEYS:

        openai.api_key = key["key1"]
        openai.api_type = "azure"
        openai.api_base = "https://" + key["name"] + ".openai.azure.com/"
        openai.api_version = "2023-03-15-preview"

        openai_objs.append(openai)

    return openai_objs

