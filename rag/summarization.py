from llm import get_llm

def llm_summary(text: str) -> str:
    """
    Generate a compact structured summary using Qwen.

    The model rewrites the provided text into a retrieval-friendly
    format containing title, summary, and keywords.

    The generated summary is used when the original grant header
    becomes too long for efficient retrieval.

    Args:
        text (str):
            Grant summary text.

    Returns:
        str:
            Structured one-line representation of the grant.
    """
    model, tokenizer = get_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a structured information extraction engine for retrieval systems. "
                "Your task is to rewrite input into a SINGLE formatted line.\n\n"
                "RULES:\n"
                "- Output ONLY one line\n"
                "- No explanations, no reasoning, no extra text\n"
                "- No markdown\n"
                "- Do NOT repeat input text\n"
                "- If keywords are missing, output empty string after Keywords:\n"
                "- Keep summary 3–6 sentences, factual and neutral\n"
                "- Maximum length: 1000 characters total"
                "Always follow exact format:\n"
                "Title: <title>\n"
                "Summary: <summary>\n"
                "Keywords: <keywords>\n"
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=180,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id
    )

    gen_tokens = outputs[0][inputs.input_ids.shape[1]:]

    result = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return result.strip()