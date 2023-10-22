import os
import time
import traceback
import sys
import pandas as pd

from dataloader.load_data import load_HealthQA, load_MedicationQA, load_LiveQA
from setup import project_setup, openai_setup
from utils_chatgpt import get_response
from const import set_constants
from argparse import ArgumentParser




def get_eval(data_df, lang, open_ai_object_list, constants):

    print("Lang: ", lang)
    

    model_use_count = 0
    model_list_index = 0

    llm_answer_list = []

    for idx, row in data_df.iterrows():
        retry = True

        if idx%100 == 0:
            print("Index: ", idx)

        while retry:
            
            try:
                
                message_list=[{'role': 'system', 'content': f'You are Health GPT and You answer to health and medical related queries in {lang}. Your answers should be in one or more paragraphs without listing points/lists and should be in {lang}.'}]
                messages = message_list.copy()
                if lang=="English":
                    prompt = prompt = row['question']
                else:
                    prompt = row['translated_question_'+lang]
                
                messages.append({'role': 'user', 'content': prompt})
                
                llm_response = get_response(open_ai_object_list[model_list_index], messages, constants['GPT_MODEL'], constants['DEPLOYMENT_ID'])
                
               
                llm_answer_list.append(llm_response)

                time.sleep(0.3)


                retry = False

                model_use_count += 1
                if model_use_count%3 == 0:
                    model_list_index += 1
                    model_list_index = model_list_index%total_models_num
                    model_use_count = 0

                    

            except Exception as e:
                print("Error at index: ", idx)
                traceback.print_exc()
                model_list_index += 1
                model_list_index = model_list_index%total_models_num
                model_use_count = 0

                print("Error: ", e)
                #check if the error contains the substring Request timed out: HTTPSConnectionPool or rate limit

                if "Request timed out: HTTPSConnectionPool" in str(e) or "rate limit" in str(e) or "timed out" or "No Response" in str(e):
                    print("Sleeping for 10 seconds")
                    time.sleep(10)
                    continue

                else:
                    if llm_response:
                        llm_answer_list.append(llm_response)
                    else:
                        if "This model's maximum context length is 8192 tokens" in str(e):
                            llm_answer_list.append("Max Context Length Exceeded")

                        else:
                            llm_answer_list.append(str(e))

                    print("LLM Response: ", llm_response)
                    retry = False
                    continue



    data_df["llm_answer_"+lang] = llm_answer_list
        
    return data_df

if __name__ == "__main__":

    #add an argument to get the dataset path
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model = args.model

    
    lang_list = ["English", "Hindi", "Chinese", "Spanish"]
    

    file_name = dataset_path.split(".")[0]
    #get the directory path from the dataset path
    directory_path = "/".join(dataset_path.split("/")[:-1])
    os.chdir(directory_path)
    print("Current working directory: {}".format(os.getcwd()))

    constants = set_constants(model)
    print(constants)
    project_setup()
    open_ai_object_list = openai_setup()
    total_models_num = len(open_ai_object_list)

    #if dataset path ends with pkl, then use pandas read_pickle method
    if dataset_path.endswith("pkl"):
        data_df = pd.read_pickle(dataset_path)
    else:

        df_li = []

        for lang in lang_list:
            if "healthqa" in args.dataset_path:
                # Only consider the dev set for HealthQA
                df = load_HealthQA(split="dev",
                                    language=lang, task="accuracy")

            elif "medicationqa" in args.dataset_path:
                df = load_MedicationQA(language=lang, task="accuracy")

            elif "liveqa" in args.dataset_path:
                df = load_LiveQA(language=lang, task="accuracy")

            else:
                raise ValueError(f"Unknown dataset {args.dataset_path}")

            if lang == "English":
                df = df[["question", "answer"]]

            else:
                df = df[["question_translated", "answer_translated"]]

                df = df.rename({
                    "question_translated": f"translated_question_{lang}",
                    "answer_translated": f"translated_answer_{lang}"
                })

            df_li += [df.reset_index(drop=True)]

        data_df = pd.concat(df_li, axis=1)

    for lang in lang_list:
        data_df = get_eval(data_df, lang, open_ai_object_list, constants)
        print("Done with language: ", lang)

    #save the dataframe as tsv file if the dataset path ends with tsv
    if dataset_path.endswith("tsv"):
        data_df.to_csv(file_name+"_with_answers.tsv", sep="\t", index=False)
    else:
        data_df.to_pickle(file_name+"_with_answers.pkl")