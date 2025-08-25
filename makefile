container_name := vllm_test
version := 0.0.0
image_name := $(container_name):$(version)

.PHONY: bulid
.PHONY: run

build:
	docker build --no-cache -t $(container_name):$(version) .
run:
	docker compose up
local:
	MODEL_PATH=../../vllm-models/Llama-3.2-1B-Instruct uv run uvicorn src.main:app --reload