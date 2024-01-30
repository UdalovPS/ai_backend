"""module with pydantic classes for validation"""

from pydantic import BaseModel
from typing import List


class CheckWorkResponseSchem(BaseModel):
    """Model for response in 'check' route"""
    info: str


class MessagesSchem(BaseModel):
    """Model for validation list of messages"""
    role: str  # system or user or assistant
    content: str


class InDataSchem(BaseModel):
    """Model for validation inner data in API"""
    messages: List[MessagesSchem]
    temperature: float
    max_tokens: int


class MLSuccessAnswer(BaseModel):
    """Model for validation success response for 'ml/answer' route"""
    text: str  # ML text answer
    token: int  # Count token for answer
    time: float  # time for generate answer with seconds


class MLAnswer(BaseModel):
    """Model for ML response for 'ml/answer' route"""
    success: bool
    data: MLSuccessAnswer
    error: str