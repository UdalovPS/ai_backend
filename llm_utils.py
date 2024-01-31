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
    parced_prompt = ""
    temperature = data.temperature
    max_tokens = data.max_tokens

    # Обработка сообщений
    for message in data.messages:
        if message.role == "system":
            parced_prompt += f"<s>system\n {message.content} </s> "
        elif message.role == "user":
            parced_prompt += f"<s>user\n {message.content} </s> "
        elif message.role == "assistant":
            parced_prompt += f"<s>assistant\n {message.content} </s> "

    parced_prompt += "<s>bot"
    return parced_prompt, max_tokens, temperature


def model_inference(parsed_prompt: str,
                    max_tokens: int,
                    temperature: float):
    logger.info(f"Request for ML model data. Text: {parsed_prompt}. Max_tokens: {max_tokens}. Temperature: {temperature}")
    logger.info(f"Create ML model with path: {MODEL_PATH_1}")
    # try:
    llm = LlamaCpp(
        model_path=MODEL_PATH_2,
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=True,
        # n_ctx=2048,       up context
        # n_gpu_layers=-1    add GPU
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
        chat_format="llama-2",
        n_gpu_layers=-1,
        n_ctx=2048
    )

    logger.info(f"Dump data: {data}")
    answer = llm.create_chat_completion(
        messages=data["messages"],
        max_tokens=in_data.max_tokens,
        temperature=in_data.temperature
    )

    logger.info(answer)

    return {"text": answer["choices"][0]["message"]["content"], "token": answer["usage"]["total_tokens"]}

