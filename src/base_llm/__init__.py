from vllm.model_executor import set_random_seed
from random import randint

from src.base_llm.multi_request_gpt import MultiRequestsGPT
from src.base_llm.gpt import GPT
from src.base_llm.llama_hf import LlamaHF
from src.base_llm.llama_vllm import LlamaVLLM
from src.base_llm.vicuna_vllm import VicunaVLLM
from src.base_llm.tulu_vllm import TuluVLLM
from src.base_llm.olmo_vllm import OLMoVLLM
from src.base_llm.vicuna_hf import VicunaHF
from src.base_llm.chatglm_hf import ChatGLMHF
from src.base_llm.gpt_request import GPTRequest
from src.base_llm.gpt4_unofficial import GPT4Uno
# from src.base_llm.qwen_api import QWenAPI
from src.base_llm.replicate_api import ReplicateAPI
# from src.base_llm.gemini_api import GeminiAPI
# from src.base_llm.baidu_api import BaiduAPI
# from src.base_llm.claude_unofficial import ClaudeUnofficialAPI
support_model_name = ["vllm-llama-2", "vllm-vicuna", "vllm-tulu", "vllm-olmo",
                      "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-4",
                      "unofficial-gpt-4-0613", "unofficial-gpt-4-1106-preview", "unofficial-gpt-3.5-turbo-1106", "unofficial-gpt-3.5-turbo-0613",
                      "request-gpt-4-0613", "request-gpt-3.5-turbo-1106", "request-gpt-3.5-turbo-0613",
                      "llama-2",
                      "vicuna",
                      "chatglm",
                      "gemini-pro",
                      "ERNIE-Bot 4.0", "ERNIE-Bot-8K", "ERNIE-Bot", "ERNIE-Bot-turbo",
                      "Qwen",
                      "replicate-meta/llama-2-7b-chat", "replicate-meta/llama-2-70b-chat", "replicate-meta/llama-2-7b", "replicate-meta/llama-2-70b"]


def load_model(model_name, loaded_model=None, loaded_tokenizer=None):
    assert model_name in support_model_name, f"Model name not in {support_model_name}."
    if "replicate" in model_name:
        lm = ReplicateAPI(model_name.replace("replicate-", ""))
    elif "request-" in model_name:
        lm = GPTRequest(model_name.replace("request-", ""))
    elif "unofficial-" in model_name:
        lm = GPT4Uno(model_name.replace("unofficial-", ""))
    elif model_name == "vllm-llama-2":
        gen_seed = randint(1, 100000)
        lm = LlamaVLLM(model_name, loaded_model, loaded_tokenizer)
        set_random_seed(gen_seed)
    elif model_name == "vllm-vicuna":
        lm = VicunaVLLM(model_name, loaded_model, loaded_tokenizer)
    elif model_name == "vllm-tulu":
        lm = TuluVLLM(model_name, loaded_model)
    elif model_name == "vllm-olmo":
        lm = OLMoVLLM(model_name, loaded_model)
    elif "gemini" in model_name:
        lm = GeminiAPI(model_name)
    elif "gpt" in model_name:
        if "gpt-4" in model_name:
            lm = GPT(model_name)
        else:
            lm = MultiRequestsGPT(model_name)
    elif "ERNIE" in model_name:
        lm = BaiduAPI(model_name)
    elif "Qwen" in model_name:
        lm = QWenAPI(model_name)
    elif 'llama-2' in model_name:
        lm = LlamaHF(model_name, loaded_model, loaded_tokenizer)
    elif 'vicuna' in model_name:
        lm = VicunaHF(model_name, loaded_model, loaded_tokenizer)
    elif 'chatglm' in model_name:
        lm = ChatGLMHF()
    return lm


if __name__ == '__main__':
    # llm = load_model("unofficial-gpt-4-0613")
    llm = load_model("vllm-olmo")
    input_list = llm.preprocess_input(["Tell me your name."])
    print(input_list)
    print(llm.batched_generate(input_list, 512, 0, 0.95))

