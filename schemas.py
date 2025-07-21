# schemas.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union


class APIRequestMessageContent(BaseModel):
    """Defines a part of a multimodal message content (e.g., text or image)."""
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class APIRequestMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str
    content: Union[str, List[APIRequestMessageContent]]


class APIRequestBody(BaseModel):
    """The main body of the request sent to the Chat Completions API."""
    model: str
    messages: List[APIRequestMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class APIResponseMessage(BaseModel):
    """The message object returned by the API."""
    content: Optional[str] = None
    role: str


class Choice(BaseModel):
    """A single choice from the list of API responses."""
    finish_reason: str
    index: int
    message: APIResponseMessage


class Usage(BaseModel):
    """Token usage statistics for the API call."""
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class APIResponseBody(BaseModel):
    """The top-level structure of the API's JSON response."""
    id: str
    created: int
    model: str
    object: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[Any] = None
    vertex_ai_grounding_metadata: Optional[List[Any]] = Field(default_factory=list)
    vertex_ai_url_context_metadata: Optional[List[Any]] = Field(default_factory=list)
    vertex_ai_safety_results: Optional[List[Any]] = Field(default_factory=list)
    vertex_ai_citation_metadata: Optional[List[Any]] = Field(default_factory=list)