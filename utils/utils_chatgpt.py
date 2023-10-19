import traceback
from typing import Union

import numpy as np
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

import const


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(prompt: str, temperature, deployment_id: Union[None, str] = None, task=const.MEDICAL) -> str:
    messages = []

    if task in [const.MEDICAL]:
        message_system = {
            'role': 'system',
            'content': 'Assistant is a medical expert that answers health-related questions.'
        }
    elif task in [const.TRANSLATE, const.PARAPHRASE]:
        message_system = {
            'role': 'system',
            'content': 'Assistant is a translator, linguist, spelling corrector and improver.'
        }

    else:
        raise NotImplementedError


    if deployment_id is None:
        response = openai.ChatCompletion.create(
            messages=[
                message_system,
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            model=const.GPT_MODEL,
            temperature=temperature,
        )
    else:
        response = openai.ChatCompletion.create(
            deployment_id=deployment_id,
            messages=[
                message_system,
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            model=const.GPT_MODEL,
            temperature=temperature,
        )

    message = response.choices[0].message
    try:
        print(message)

    except:
        print("Error printing message")
        traceback.print_exc()

    if hasattr(message, "content"):
        return message.content
    else:
        return np.NaN


