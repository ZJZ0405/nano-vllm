import argparse
import time
from dataclasses import fields
from typing import Any, Literal, TypedDict, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nanovllm import LLM, SamplingParams
from nanovllm.config import Config


class GenerationOutput(TypedDict):
    text: str
    token_ids: list[int]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nanovllm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: None = None
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class OpenAIServer:
    def __init__(self, model: str, **llm_kwargs: Any):
        self.model = model
        self.created = int(time.time())
        self.llm = LLM(model, **llm_kwargs)

    def sampling_params(self, request: CompletionRequest | ChatCompletionRequest) -> SamplingParams:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming responses are not supported")
        try:
            return SamplingParams(
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                ignore_eos=request.ignore_eos,
            )
        except AssertionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def encode_len(self, text: str) -> int:
        return len(self.llm.tokenizer.encode(text))

    def apply_chat_template(self, messages: list[ChatMessage]) -> str:
        message_dicts = [message.dict() for message in messages]
        return self.llm.tokenizer.apply_chat_template(
            message_dicts,
            tokenize=False,
            add_generation_prompt=True,
        )


def create_app(model: str, **llm_kwargs: Any) -> FastAPI:
    server = OpenAIServer(model, **llm_kwargs)
    app = FastAPI(title="nano-vLLM OpenAI API Server")
    app.state.openai_server = server

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelList)
    def list_models() -> ModelList:
        return ModelList(data=[ModelCard(id=server.model, created=server.created)])

    @app.post("/v1/completions", response_model=CompletionResponse)
    def create_completion(request: CompletionRequest) -> CompletionResponse:
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        sampling_params = server.sampling_params(request)
        outputs = cast(list[GenerationOutput], server.llm.generate(prompts, sampling_params, use_tqdm=False))
        choices = [
            CompletionChoice(index=index, text=output["text"])
            for index, output in enumerate(outputs)
        ]
        prompt_tokens = sum(server.encode_len(prompt) for prompt in prompts)
        completion_tokens = sum(len(output["token_ids"]) for output in outputs)
        return CompletionResponse(
            id=f"cmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model or server.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        prompt = server.apply_chat_template(request.messages)
        sampling_params = server.sampling_params(request)
        output = cast(list[GenerationOutput], server.llm.generate([prompt], sampling_params, use_tqdm=False))[0]
        prompt_tokens = server.encode_len(prompt)
        completion_tokens = len(output["token_ids"])
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model or server.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(content=output["text"]),
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve nano-vLLM with an OpenAI-compatible HTTP API.")
    parser.add_argument("--model", required=True, help="Path to the model directory.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level.")

    config_fields = {field.name: field for field in fields(Config) if field.name != "model"}
    for name, field_info in config_fields.items():
        default = field_info.default
        if isinstance(default, bool):
            parser.add_argument(f"--{name.replace('_', '-')}", action="store_true", default=default)
        elif isinstance(default, int):
            parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_kwargs = {
        field.name: getattr(args, field.name)
        for field in fields(Config)
        if field.name != "model" and hasattr(args, field.name)
    }
    app = create_app(args.model, **llm_kwargs)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
