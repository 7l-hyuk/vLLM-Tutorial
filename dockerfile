FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu20.04

WORKDIR /app

RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y python3 python3-pip ray

COPY . .

RUN uv sync

VOLUME ["/app/models"]

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]