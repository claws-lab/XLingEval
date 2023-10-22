def prompt_translate(sentence, source_lang, target_lang, n_shots=0):
    if n_shots == 0:

        prompt = f"Translate the following from {source_lang} to {target_lang}. The input is delimited with triple backticks. \nInput: ```{sentence}```\nOutput:"

    else:
        raise NotImplementedError("Multi-shot translation is not implemented yet.")

    return prompt