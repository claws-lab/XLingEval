def prompt_consistency(question: str, language: str, use_bullet_point: bool) -> str:

    if language == "Chinese":
        infix_bullet_point = "你的回答应该用无序列表罗列，每条内容应以'-'开头。" if use_bullet_point else ""

        prompt = f"我想让你扮演一个医生，回答下列医学问题，或根据下列情境提出诊疗方案。{infix_bullet_point}\n问题: ```{question}```"


    elif language == "English":

        infix_bullet_point = "Write your answer in a bullet-by-bullet format. Each line should start with a '-'." if use_bullet_point else ""

        prompt = f"I want you to act as a doctor and answer the following medical question, or give your medical diagnosis based on the following scenario. {infix_bullet_point}\nQuestion: ```{question}```"

    elif language == "Chinese-Traditional":

        infix_bullet_point = "你的回答應該用無序列表羅列，每條內容應以'-'開頭。" if use_bullet_point else ""

        prompt = f"我想讓你扮演一個醫生，回答下列醫學問題。{infix_bullet_point}\n問題：```{question}```"

    elif language == "Hindi":
        infix_bullet_point = f"अपने उत्तर लाइन-बाय-लाइन लिखें। हर लाइन '-' से शुरू होनी चाहिए" if use_bullet_point else ""

        prompt = f"मैं चाहता हूं कि आप एक डॉक्टर के रूप में कार्य करें और निम्नलिखित चिकित्सा प्रश्न का उत्तर दें, या निम्नलिखित परिदृश्य के आधार पर अपना चिकित्सा निदान दें। {infix_bullet_point}\nप्रश्न: ```{question}```"

    elif language == "Spanish":
        infix_bullet_point = "Escribe tu respuesta viñeta por viñeta. Cada línea debe empezar por '-'." if use_bullet_point else ""

        prompt = f"Quiero que actúes como un médico y respondas a la siguiente pregunta médica, o des tu diagnóstico médico basado en el siguiente escenario. {infix_bullet_point}\nPregunta: ```{question}``."

    else:
        raise NotImplementedError


    return prompt

