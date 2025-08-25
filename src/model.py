from vllm import EngineArgs, LLMEngine, SamplingParams
from .config import MODEL_PATH

engine_args = EngineArgs(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.75,
    dtype="float16",
    max_model_len=1536,
    enforce_eager=True,
    max_num_seqs=1,
    max_num_batched_tokens=512,
    swap_space=6
)

llm = LLMEngine.from_engine_args(engine_args)
sampling_params = SamplingParams(
    temperature=0.0,
    top_k=1,
    max_tokens=1024
)
