"""main module. App start here"""

import logging
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime

# schemas imports
from schemas import CheckWorkResponseSchem, InDataSchem, MLSuccessAnswer, MLAnswer

# import global variables
from response_core import SuccessResponse, ErrorResponse

# import llm logic
from llm_utils import parse_json, model_inference, create_answer_from_ml

# set logging level
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# create FastApi object
app = FastAPI(
    title='AI app',
    docs_url="/ml_app/docs"  # swagger open to .../ai_app/docs
)

PREFIX = "/ml_app"  # prefix for security (need for nginx)


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
            content=response_data.response.model_dump()  # transform pydantic to dict
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
        start_time = datetime.now()  # detect start time
        # p, mt, t = parse_json(in_data)
        # res = model_inference(p, mt, t)
        res = create_answer_from_ml(in_data=in_data)
        answer_time = datetime.now() - start_time  # detect language model work time
        response_data = SuccessResponse(
            data=MLSuccessAnswer(
                text=res['text'],
                token=res['token'],
                time=answer_time.total_seconds()
            ),
            api="ml"
        )
    except Exception as ex:
        logger.warning(f"Error to generate ML answer: {ex}")
        logger.exception(ex)
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
            content=response_data.response.model_dump()  # transform pydantic to dict
        )
