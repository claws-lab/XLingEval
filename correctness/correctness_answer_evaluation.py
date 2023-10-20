import os
import time
import traceback
import sys
import pandas as pd
from setup import project_setup, openai_setup
from utils_chatgpt import get_response
from const import set_constants
from argparse import ArgumentParser


def prompt_type(question, answer_1, answer_2, lang):
    case_1_string = f"Answer 2 provides neither contradictory nor similar information in comparison to Answer 1"
    case_2_string = f"Answer 2 provides contradictory information in comparison to Answer 1"
    case_3_string = f"Answer 2 provides more comprehensive and appropriate information"
    case_4_string = f"Answer 2 provides less comprehensive and appropriate information"
    #prompt = f"Given two answers written in {lang} for a particular question, Answer 1 provides the answer from dataset and Answer 2 is the candidate answer. Please evaluate the comphrehensiveness and appropriateness of Answer 2 as compared to Answer 1 on the basis of how similar they are and if Answer 2 is more comprehensive and appropriate. First report just the answer number for the answer that is more comprehensive and appropriate and then always provide the reasoning for choosing the answer. Don't provide any answer without providing reasoning for it.\nAnswer 1: {answer_1}\nAnswer 2: {answer_2}\n. Lastly, just provide one label based on the following cases:{case_6_string}\n{case_7_string}\n{case_8_string}\n. The label should provided in the format Correct Label: <label_number>"
    contradiction_info_string = f"Answer 2 contradicts Answer 1 if Answer 2 provides any information that contradicts the information provided in Answer 1 and Answer 2 incorrectly answers the Question. Answer 2 is neither contradictory nor similar if Answer 2 answers the Question but Answer 1 and Answer 2 are dissimilar."
    output_structure_string = f"Your answer should first provide all reasoning followed by the choosen option string in the last line. The chosen option string should exactly match with one of the given options."
    prompt = f"Given below is the question and two answers written in {lang} for the question.\nQuestion: {question}\nAnswer 1: {answer_1}\nAnswer 2: {answer_2}\nCompare Answer 2 with Answer 1 on the basis of answer similarity, comprehenisveness and appropriateness. {contradiction_info_string}\nFirst, evaluate whether Answer 2 provides similar information, contradictory information as compared to Answer 1. Support the evaluation with a reasoning. Only if both answers are similar, evaluate the comprehensivenss and appropriateness of Answer 2 in comparison to Answer 1 and provide a reasoning for it. Finally, based on the previous evaluations choose one option from the following option:\n1) {case_1_string}\n2) {case_2_string}\n3) {case_3_string}\n4) {case_4_string}\n{output_structure_string}"
    
    return prompt


def get_eval(data_df, lang, open_ai_object_list, constants):

    print("Lang: ", lang)

    case_1_string = f"Answer 2 provides neither contradictory nor similar information in comparison to Answer 1"
    case_2_string = f"Answer 2 provides contradictory information in comparison to Answer 1"
    case_3_string = f"Answer 2 provides more comprehensive and appropriate information"
    case_4_string = f"Answer 2 provides less comprehensive and appropriate information"
    

    model_use_count = 0
    model_list_index = 0

    llm_answer_eval_list = []
    case_label_list = []

    
    option_string_list = [case_1_string, case_2_string, case_3_string, case_4_string]


    for idx, row in data_df.iterrows():
        retry = True

        if idx%100 == 0:
            print("Index: ", idx)

        while retry:
            
            try:
                
                #select one column between answer and question_answered randomly
                message_list = [{"role": "system", "content": "You are Health GPT and you help users in the health and medical related domain."}]
                messages = message_list.copy()
                if lang=="English":
                    prompt = prompt_type(row['question'], row['answer'], row['llm_answer_'+lang], lang)
                else:
                    prompt = prompt_type(row['translated_question_'+lang], row['translated_answer_'+lang], row['llm_answer_'+lang], lang)
                
                messages.append({'role': 'user', 'content': prompt})
                
                llm_response = get_response(open_ai_object_list[model_list_index], messages, constants['GPT_MODEL'], constants['DEPLOYMENT_ID'])
                case_label = ""
                

                if llm_response == "":
                    case_label = "No Response"
                
                else:
                
                    #check if any of the string present in the option_string_list is present in the response if yes then assign that string to case_label_English
                    llm_response_check_list = (" ".join(llm_response.split("\n"))).split(".")
                    #get the last two sentences from the response
                    llm_response_check = " ".join(llm_response_check_list[-3:])
                    
                    for option_string in option_string_list:
                        if option_string in llm_response_check:
                            case_label = option_string
                            break
                    
                    if case_label == "":
                        case_label = "No Response"
                
                time.sleep(0.3)

                #print("LLM Response: ", llm_response)
                #print("Case Label: ", case_label)

                llm_answer_eval_list.append(llm_response)
                case_label_list.append(case_label)
                #code for saving intermediate files

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
                    #check if llm_response_English exists and is not empty
                    if llm_response:
                        llm_answer_eval_list.append(llm_response)
                        case_label_list.append(case_label)
                    else:
                        if "This model's maximum context length is 8192 tokens" in str(e):
                            llm_answer_eval_list.append("Max Context Length Exceeded")
                            case_label_list.append("Max Context Length Exceeded")

                        else:
                            llm_answer_eval_list.append(str(e))
                            case_label_list.append(str(e))

                    print("LLM Response: ", llm_response)
                    print("Case Label: ", case_label)
                    retry = False
                    continue



    data_df["llm_answer_eval_"+lang] = llm_answer_eval_list
    data_df["case_label_"+lang] = case_label_list
        
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
    data_df = pd.read_csv(dataset_path, sep="\t")
    

    for lang in lang_list:
        data_df = get_eval(data_df, lang, open_ai_object_list, constants)
        print("Done with language: ", lang)

    #save the dataframe as tsv file
    data_df.to_csv(file_name+"_prompt_evaluation.tsv", sep="\t", index=False)


