from typing import Union

import openai


def get_response(open_ai_object, message_list:list, model: str, deployment_id: Union[None, str] = None) -> str:
    messages = []

    if deployment_id is None:
        response = openai.ChatCompletion.create(
            messages=messages,
            model=model,
            temperature=0,
        )
    else:
        #print("Messages: ", message_list)
        response = open_ai_object.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=message_list,
            model=model,
            temperature=0,
            max_tokens=1000

        )
    if 'content' in response.choices[0].message:
        if response.choices[0].message.content is not None:
            return response.choices[0].message.content
        else:
            return "None Response"
    else:
        return "No Response"
