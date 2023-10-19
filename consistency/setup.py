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
]


def set_seed(seed, use_torch=True):
    import random
    import numpy as np

    import warnings
    import pandas as pd
    warnings.simplefilter(action='ignore', category=FutureWarning)

    check_cwd()

    print(f"Setting seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)

    if use_torch:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def openai_setup(args):
    import openai
    project_name = KEYS[args.idx_auth]['name']

    openai.api_version = '2023-03-15-preview'
    openai.api_base = f'https://{project_name}.openai.azure.com/'

    openai.api_type = 'azure'

    openai.api_key = KEYS[args.idx_auth]["key1"]
