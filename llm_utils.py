import json
import time
from langchain_community.llms import LlamaCpp
import re
import logging


from langchain.callbacks.manager import CallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_cpp import Llama

from config import MODEL_PATH_1, MODEL_PATH_2
from schemas import InDataSchem

logger = logging.getLogger(__name__)


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
    logger.info(f"Request for ML model data. Text: {parsed_prompt}. Max_tokens: {max_tokens}. Temperature: {temperature}")
    logger.info(f"Create ML model with path: {MODEL_PATH_1}")
    # try:
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

        logger.info(f"ML res. Text: {res}, Tokens: {token_count}")
        return {"text": res, "token": token_count}

    else:
        logger.error(f"Ml model not generate answer")
        return {"text": None, "token": None}


def create_answer_from_ml(in_data: InDataSchem):
    logger.info(f"Received data to generate a response")
    logger.debug(f"Inner data: {in_data}")
    data = in_data.model_dump()
    llm = Llama(
        model_path=MODEL_PATH_1,
        verbose=True,  # Verbose is required to pass to the callback manager,
        chat_format="llama-2"
    )

    logger.info(f"Dump data: {data}")
    answer = llm.create_chat_completion(
        messages=data["messages"],
        max_tokens=in_data.max_tokens,
        temperature=in_data.temperature
    )

    logger.info(answer)

    return {"text": answer["choices"][0]["message"]["content"], "token": answer["usage"]["total_tokens"]}

