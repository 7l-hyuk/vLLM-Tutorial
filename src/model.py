from vllm import LLM, SamplingParams
from .config import MODEL_PATH

llm = LLM(
    model=MODEL_PATH,
    dtype="float16",
    tensor_parallel_size=1,
    max_model_len=1536,
    gpu_memory_utilization=0.75,
    enforce_eager=True,
    max_num_seqs=1,
    max_num_batched_tokens=512,
    swap_space=6,
)
sampling_params = SamplingParams(
    temperature=0.0,
    top_k=1,
    max_tokens=1024
)
