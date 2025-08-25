import os
import asyncio
import logging
from uuid import uuid4

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from vllm import SamplingParams

from .model import llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logging.info(f"MODEL = {os.getenv('MODEL_PATH')}")

app = FastAPI()

class QueryRequest(BaseModel):
    request_id: str
    query: str

    n: int = Field(default=1)
    top_p: float = Field(default=1)
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=1024)
    seed: int = Field(default=42)


@app.post("/generate")
async def generate_response(request: QueryRequest):
    sent_text = ""  # 생성된 text 저장

    async def stream_response():
        nonlocal sent_text

        sampling_params = SamplingParams(
            n=request.n,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=1.1,
            max_tokens=request.max_tokens,
            seed=request.seed
        )
        result_generator = llm.generate(
            request.query,
            sampling_params,
            request_id=request.request_id
        )

        async for output in result_generator:
            text = output.outputs[0].text
            send_text = text[len(sent_text):]
            sent_text = text

            for word in send_text.split(" "):
                if word:
                    yield word + " "
            
            if output.finished:
                return
    
    return StreamingResponse(stream_response(), media_type="text/plain")
