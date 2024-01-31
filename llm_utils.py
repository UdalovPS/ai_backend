import json
import time
from langchain_community.llms import LlamaCpp
import re

from config import MODEL_PATH_1, MODEL_PATH_2
from schemas import InDataSchem


def parse_json(data: InDataSchem):
    """
    Function parser for upcoming jsons
    :param data: json dict with structure:
        in_data: InDataSchem(
            messages:[
                MessagesSchem(
                    role: (str) - ("system" or "assistant" or "user")
                    content: (str) - text of message
                )
                ...
            ]
            temperature: (float) - ML model variable
            max_tokens: (int)   - token count for answer
        )
    :return: ( parced_prompt,
               max_tokens,
               temperature )
    """
    # Extracting the system prompt
    # system_prompt = data["messages"][0]["content"] if data["messages"][0]["role"] == "system" else "No system prompt"
    system_prompt = data.messages[0].content
    temperature = data.temperature
    max_tokens = data.max_tokens
    # Extracting temperature and max_tokens
    # temperature = data["temperature"]
    # max_tokens = data["max_tokens"]

    # Creating the dialog string
    dialog = "Диалог:\n"
    # for message in data["messages"][1:]:
    #     dialog += f"{message['role']} - {message['content']}\n"
    for message in data.messages[1:]:
        dialog += f"{message.role} - {message.content}\n"

    # Formatting the final output
    parced_prompt = f"{system_prompt}\n\n{dialog}\nТвой ответ:"
    return parced_prompt, max_tokens, temperature


def model_inference(parsed_prompt: str,
                    max_tokens: int,
                    temperature: float):
    response = {
        "success": False,
        "data": {"text": None,
                 "token": None},
        "error": None
    }

    try:
        llm = LlamaCpp(
            model_path=MODEL_PATH_1,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=False
        )
        res = llm.invoke(parsed_prompt)
        # Проверка, есть ли результат от нейросети
        if res:
            # Разделение строки на токены
            tokens = re.findall(r'\b\w+\b', res)
            token_count = len(tokens)

            response["success"] = True
            response["data"]["text"] = res
            response["data"]["token"] = token_count

        else:
            response["error"] = "Нейросеть не вернула результат"

    except Exception as e:
        response["error"] = str(e)

    return response
