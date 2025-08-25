from uuid import uuid4
import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .model import llm, sampling_params

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/generate")
async def generate_response(request: QueryRequest):
    request_id = str(uuid4())   # 요청 구분용 uuid
    llm.add_request(request_id, request.query, sampling_params)
    sent_text = ""  # 생성된 text 저장

    async def stream_response():
        nonlocal sent_text
        while True:
            request_outputs = llm.step()

            for output in request_outputs:
                if output.request_id == request_id:
                    text = output.outputs[0].text
                    send_text = text[len(sent_text):]   # 이미 받은 text 이후부터
                    sent_text = text                    # sent_text 업데이트

                    for word in send_text.split(" "):   # 띄어쓰기 단위로 streaming
                        if word:
                            yield word + " "
                    if output.finished:
                        return
    
            await asyncio.sleep(0.1)
    
    return StreamingResponse(stream_response(), media_type="text/plain")
