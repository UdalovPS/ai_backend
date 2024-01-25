"""module for unified response from API"""

from typing import Any
from pydantic import BaseModel


class BasePydanticResponse(BaseModel):
    """Base pydantic class for unified response"""
    success: bool
    api: str
    data: Any       # if fact this attr is pydantic object or None
    error: Any      # if fact this attr is pydantic object or None


class BaseResponse:
    """Base interface class. Specifies response format"""
    def __init__(self, status_code: int, success: bool, api: str,
                 data: [BaseModel, None], error: [BaseModel, None]):
        self.status_code = status_code      # HTTP status code
        self.response = BasePydanticResponse(
            success=success,
            api=api,
            data=data,
            error=error
        )


class SuccessResponse(BaseResponse):
    """Class for success response from API"""
    def __init__(self, data: BaseModel, api: str, status_code=200, success=True, error=None):
        """
        Args:
            data: any custom pydantic object with data of successful response
            api: identified API (any string value)
            status_code: HTTP status code
            success: bool flag (in success = True)
            error: any custom pydantic object with data of error response (in success = None)
        """
        super().__init__(status_code=status_code, success=success, data=data, error=error, api=api)


class ErrorResponse(BaseResponse):
    """Class for error response from API"""
    def __init__(self, error: BaseModel, api: str, status_code: int, success=False, data=None):
        """
        Args:
            data: any custom pydantic object with data of successful response (in error = None)
            api: identified API (any string value)
            status_code: HTTP status code
            success: bool flag (in error = False)
            error: any custom pydantic object with data of error response
        """
        super().__init__(status_code=status_code, success=success, data=data, error=error, api=api)
