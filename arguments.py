import argparse
import os.path as osp
import platform

import const


if platform.system() == 'Windows':
    REDDIT_COMMENTS_DIR = "E:\\data\\Reddit\\comments"
    DATA_DIR = "F:\\data\\NLP"
    DEVICE_MAP = {"": 0}

elif platform.system() == 'Linux':

    if osp.exists(const.HOME_DIR_LINUX_SERVER):
        DATA_DIR = osp.join(const.HOME_DIR_LINUX_SERVER, "Workspace", "data", "NLP")
        DEVICE_MAP = {"": [0, 1, 2, 3]}

    elif osp.exists(const.HOME_DIR_LINUX):
        DATA_DIR = osp.join(const.HOME_DIR_LINUX, "Workspace", "storage", "NLP")
        DEVICE_MAP = {"": [0, 1]}

    else:
        raise ValueError("Unknown system.")

elif platform.system() == 'Darwin':
    DATA_DIR = "data"
    DEVICE_MAP = {"": 0}

else:
    raise ValueError("Unknown system.")

parser = argparse.ArgumentParser(description="")
# Parameters for Analysis
parser.add_argument('--batch_size', type=int,
                    default=64,
                    help="Batch sizes")
parser.add_argument('--data_dir', type=str,
                    default=DATA_DIR,
                    help="Location to store the processed dataset")
parser.add_argument('--dataset_name', type=str,
                    default="healthqa",
                    help="Which dataset to use")
parser.add_argument('--do_visual', action='store_true',
                    help="Whether to do visualization")
parser.add_argument('--do_batch', action='store_true',
                    help="Whether to do batch processing")
parser.add_argument('--debug', action='store_true', help="do debugging")
parser.add_argument('--fill_null_values', action='store_true',
                    help="Fill null values when the dataframe already exists. This is commonly used in translation.")
parser.add_argument('--device', type=str, default='cuda:0',
                    help="Device to use. When using multi-gpu, this is the 'master' device where all operations are performed.")
parser.add_argument('--idx_auth', type=int, default=0,
                    help="Index to end processing.")
parser.add_argument('--idx_end', type=int, default=-1,
                    help="Index to end processing.")
parser.add_argument('--idx_start', type=int, default=0,
                    help="Index to start processing.")

parser.add_argument('--interval', type=int, default=1, help="Interval.")
parser.add_argument('--label', type=str, default=const.POSITIVE,
                    choices=[const.POSITIVE, const.NEGATIVE],
                    help="Do we experiment with the postive or randomly sampled negative answers?")
parser.add_argument('--model', type=str, choices=["gpt35", "gpt4", "llama2",
                                                  "medalpaca-7b",
                                                  "medalpaca-13b",
                                                  "medalpaca-30b",
                                                  ],
                    required=True, help="")
parser.add_argument('--neg_sampling_ratio', type=int, default=4,
                    help="How many negative samples per positive example.")
parser.add_argument('--num_workers', type=int, default=15)
parser.add_argument('--num_answers', type=int, default=10,
                    help="How many different answers we want ChatGPT to return.")
parser.add_argument('--model_dir', type=str, default="F:\\verifiability")
parser.add_argument('--output_dir', type=str, default="outputs",
                    help="Outputs directory.")

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--source_language', type=str,
                    default='English',
                    choices=['Chinese', 'Spanish', "Hindi", "English"],
                    help="Language to translate from.")
parser.add_argument('--split', type=str, choices=['train', 'dev', 'test'],
                    default='dev',
                    help="Dataset split to experiment with.")
parser.add_argument('--target_language', type=str,
                    default='Chinese',
                    choices=['Chinese', 'Spanish', "Hindi", "English"],
                    help="Language to translate to.")
parser.add_argument('--task', type=str,
                    default=const.MEDICAL,
                    choices=[const.MEDICAL, const.TRANSLATE, const.PARAPHRASE],
                    help="Language to translate to.")
parser.add_argument('--temperature', type=float, default=0., help="")
parser.add_argument('--verbose', action='store_true',
                    help="Whether to log everything")


args = parser.parse_args()
args.use_numpy = False
args.device_map = DEVICE_MAP
