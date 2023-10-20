import os
import os.path as osp
import platform
import sys
import const
import traceback
import pandas as pd

from argparse import ArgumentParser

import re
import string
import gc

from medalpaca.inferer import Inferer

from medalpaca.params_medalpaca import *


# sys.path.append(osp.join(os.getcwd(), "consistency"))

# from consistency import statistical_test



def load_model(model_name):

    # --- Flags from the original code ---
    load_in_8bit = False

    prompt_template = f"medalpaca_accuracy_prompt.json"
    
    if model_name == "medalpaca-30b":

        base_model = "decapoda-research/llama-30b-hf"
        model_name_path = "medalpaca/medalpaca-lora-30b-8bit"
        peft = True
        
    elif model_name == "medalpaca-13b":
        base_model = "decapoda-research/llama-13b-hf"
        model_name_path = "medalpaca/medalpaca-lora-13b-8bit"
        peft = True
        
    elif model_name == "medalpaca-7b":
        
        base_model = None
        model_name_path = "medalpaca/medalpaca-7b"
        peft = True

    print("peft", peft)
    print("load_in_8bit", load_in_8bit)

    if platform.system() == "Linux":
        model = Inferer(
            model_name=model_name_path,
            prompt_template=prompt_template,
            base_model=base_model,
            peft=peft,
            load_in_8bit=load_in_8bit,
            args=args,
        )

    return model


def accuracy_medalpaca(data_df, model, batch_size, target_language):




    sampling['temperature'] = 0.001
    sampling['max_new_tokens'] = '''<enter the number of tokens you want to generate>'''

    if target_language == 'English':
        question_column = 'question'
        answer_column = 'answer'

    else:
        question_column = 'translated_question_' + target_language
        answer_column = 'translated_answer_' + target_language

    llm_answer_column = 'llm_answer_' + str(target_language)

    # iterate over the dataframe

    llm_answer_list = []
    for index in range(0, data_df.shape[0], batch_size):

        retry_flag = True
        while retry_flag:
            print("index", index)
            if index % 10 == 0:
                print(f"Processing {index}th question")

            if index + batch_size < data_df.shape[0]:
                question_list = data_df[question_column][index:index+batch_size].to_list()
            else:
                question_list = data_df[question_column][index:data_df.shape[0]].to_list()

            
            try:

                responses = model.batch_inference(
                    instruction=f"You are Health GPT and You answer to health and medical related queries in {target_language}. Your answers should be in one or more paragraphs without listing points/lists and should be in {target_language}.",
                    inputs=question_list,
                    output="Output:",
                    verbose=False,
                    **sampling
                )

                for j, response in enumerate(responses):
                    llm_answer_list.append(response)
                
                retry_flag = False
            
            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Error occured, retrying...")
    
    data_df[llm_answer_column] = llm_answer_list

    return data_df


if __name__ == "__main__":

    cuda_path = "/usr/local/cuda-11.7/bin/nvcc"
    if "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] += f"{cuda_path}"
    else:
        os.environ["LD_LIBRARY_PATH"] = cuda_path

    # project_setup(args)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", help="batch size")
    parser.add_argument("--model", help="model name")
    parser.add_argument("--dataset_path", help="path to the dataset")

    language_list = ["English", "Chinese", "Hindi", "Spanish"]

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    model_name = str(args.model)
    dataset_path = str(args.dataset_path)

    file_name = dataset_path.split(".")[0]

    model = load_model(model_name)

    data_df = pd.read_csv(dataset_path, sep='\t')
    #only keep the question, answer, translated_question_Hindi, translated_answer_Hindi,translated_question_Spanish, translated_answer_Spanish, translated_question_Chinese, translated_answer_Chinese columns
    data_df = data_df[['question', 'answer', 'translated_question_Hindi', 'translated_answer_Hindi', 'translated_question_Spanish', 'translated_answer_Spanish', 'translated_question_Chinese', 'translated_answer_Chinese']]

    for language in language_list:
        data_df = accuracy_medalpaca(data_df, model, batch_size, language)
        print(f"Done with {language} for data df")
        data_df.to_csv(file_name + '_with_answers.tsv', sep='\t', index=False)
        gc.collect()
        torch.cuda.empty_cache()
