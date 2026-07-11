from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print(torch.cuda.is_available())
model_id = "Qwen/Qwen3.5-0.8B"

_model = None
_tokenizer = None

def get_llm():
    """
    Load and return the Qwen LLM and tokenizer.

    The model is initialized only once and reused throughout
    the application using a singleton-like pattern.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]:
            Loaded model and tokenizer.
    """
    global _model
    global _tokenizer

    if _model is None:
        print("Loading model...")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto")

    return _model, _tokenizer