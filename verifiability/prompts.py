

def prompt_verifiability(question: str, response: str, language:str):

    if language == "Chinese":
        prompt = f"用一个英文单词（`Yes`或`No`，不要回答原因）告诉我下列答案是否是以下问题（或情景）的正确回答（或诊疗方案）。问题和答案均用三个反引号包围。 \n问题: ```{question}``` \n答案: ```{response}```.你的判断："

    elif language == "Hindi":
        prompt = f"मुझे उत्तर देने के लिए एक अंग्रेजी शब्द ('Yes' या 'No', कोई कारण न बताएं) का उपयोग करें कि क्या निम्नलिखित प्रतिक्रिया निम्नलिखित प्रश्न (या परिदृश्य) का तथ्यात्मक रूप से सही उत्तर (या निदान) है। प्रश्न और उत्तर दोनों को ट्रिपल बैकटिक्स द्वारा सीमांकित किया गया है।\nप्रश्न: ```{question}``` \nआपका उत्तर: ```\n{response}\n```।"

    elif language == "English":
        prompt = (f"Use one English word (either 'Yes' or 'No'. Do not give any reasons) to answer me whether the "
                  f"following response is a factually correct answer (or diagnosis) to the following question (or "
                  f"scenario). Both the question and answer are delimited by triple backticks. \nQuestion: \n```"
                  f"\n{question}\n```\nResponse: ```\n{response}\n```")

    elif language == "Spanish":
        prompt = f"Use una palabra en inglés (ya sea 'Yes' o 'No'. No dé ninguna razón) para responderme si la siguiente respuesta es una respuesta (o diagnóstico) objetivamente correcta a la siguiente pregunta (o escenario). Tanto la pregunta como la respuesta están delimitadas por triples acentos graves. \nPregunta: \n```\n{question}\n```\n \nRespuesta: ```\n{response}\n```"

    else:
        raise NotImplementedError
    return prompt



