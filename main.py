"""main module. App start here"""

import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime

# schemas imports
from schemas import CheckWorkResponseSchem, InDataSchem, MLSuccessAnswer

# import global variables
from response_core import SuccessResponse, ErrorResponse

# set logging level
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# create FastApi object
app = FastAPI(
    title='AI app',
    docs_url="/ml_app/docs"     # swagger open to .../ai_app/docs
)

PREFIX = "/ml_app"      # prefix for security (need for nginx)


@app.get(f"{PREFIX}/check")
async def check_api_work():
    """Method for check server work"""
    try:
        logger.info("Received a request to check API work")
        response_data = SuccessResponse(data=CheckWorkResponseSchem(info="ML api is work"), api="ml_api")
    except Exception as ex:
        logger.warning(f"Error to check API work: {ex}")
        response_data = ErrorResponse(error=CheckWorkResponseSchem(info="ML api error"), api="ml_api", status_code=500)
    finally:
        return JSONResponse(
            status_code=response_data.status_code,
            content=response_data.response.model_dump()     # transform pydantic to dict
        )


# START LOGIC ROUTE
@app.post(f"{PREFIX}/ml/answer")
async def get_answer_for_ml_model(in_data: InDataSchem):
    """This method get request with chat history and response language model answer
    Args:
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
    """
    try:
        logger.info(f"Received request with data: {in_data}")
        start_time = datetime.now()     # detect start time

        # EXAMPLE CODE START
        # if you need data from request body, you need use var in_data
        temperature = in_data.temperature
        token = in_data.max_tokens
        messages = in_data.messages     # this is list. You can use cycle
        for message in messages:
            role = message.role
            print(role)
            content = message.content
            print(content)
        # EXAMPLE CODE END

        # IN THIS LINE WE GET LANGUAGE MODEL ANSWER

        answer_time = datetime.now() - start_time   # detect language model work time
        response_data = SuccessResponse(
            data=MLSuccessAnswer(
                text="This is generated answer",    # replace on answer text
                token=20,   # replace on answer token
                time=answer_time.total_seconds()
            ),
            api="ml"
        )
    except Exception as ex:
        logger.warning(f"Error to generate ML answer: {ex}")
        response_data = ErrorResponse(
            error=CheckWorkResponseSchem(
                info="error to generate ML answer"
            ),
            api="ai_api",
            status_code=500
        )
    finally:
        return JSONResponse(
            status_code=response_data.status_code,
            content=response_data.response.model_dump()     # transform pydantic to dict
        )
