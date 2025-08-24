from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import RequestOutput
from .model import llm, sampling_params

app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/generate")
def generate_response(request: QueryRequest):
    try:
        res: list[RequestOutput] = llm.generate(request.query, sampling_params)
        return {"response": res[0].outputs[0].text}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
