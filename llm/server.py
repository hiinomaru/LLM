from fastapi import FastAPI
from pydantic import BaseModel

from .qwen_loader import get_llm


app = FastAPI()


# Загружаем модель один раз при старте сервера
model, tokenizer = get_llm()


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    # Старый способ через prompt
    prompt: str | None = None

    # Новый способ через chat messages
    messages: list[Message] | None = None

    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False


class GenerateResponse(BaseModel):
    answer: str


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):

    # Поддержка нового chat API
    if req.messages:

        messages = [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in req.messages
        ]

    # Поддержка старого prompt API
    elif req.prompt:

        messages = [
            {
                "role": "user",
                "content": req.prompt
            }
        ]

    else:

        return GenerateResponse(
            answer="No prompt or messages provided"
        )


    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        chat,
        return_tensors="pt"
    ).to(model.device)


    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        do_sample=req.do_sample,
        pad_token_id=tokenizer.eos_token_id
    )


    answer = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()


    return GenerateResponse(answer=answer)