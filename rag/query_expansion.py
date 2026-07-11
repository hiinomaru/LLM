from llm import get_llm

model, tokenizer = get_llm()

def llm_expand_query(query: str) -> str:
    """
    Expand a user query for grant retrieval using Qwen.

    The function rewrites a short user query into a keyword-dense query
    optimized for semantic vector search.

    The expansion process:
        - adds relevant funding-related terminology
        - maps concepts to EU research funding vocabulary
        - introduces additional retrieval signals
        - avoids natural-language explanations

    The generated query is intended to improve recall when searching
    the ChromaDB vector database.

    Args:
        query (str):
            Original user query.

    Returns:
        str:
            Expanded retrieval query suitable for embedding search.
    """

    messages = [
    {
        "role": "system",
        "content": (
            "You are a query expansion engine for a Retrieval-Augmented Generation (RAG) system focused on EU research and innovation grants.\n"
            "\n"
            "Your task is to convert a short user query into a retrieval-optimized search query for a chunked vector database.\n"
            "\n"
            "DATABASE CONTEXT:\n"
            "- EU funding programs (Horizon Europe, ERC, Marie Skłodowska-Curie, national research grants)\n"
            "- grant calls, eligibility rules, deadlines, funding schemes\n"
            "- short text chunks with partial information\n"
            "\n"
            "CRITICAL RULES:\n"
            "- Output ONLY ONE single search query\n"
            "- Do NOT output full sentences\n"
            "- Do NOT write explanations or formatting\n"
            "- Do NOT include punctuation-heavy natural language\n"
            "- Do NOT hallucinate real grant facts\n"
            "- Do NOT repeat the same meaning in different words\n"
            "\n"
            "QUERY FORMAT REQUIREMENTS:\n"
            "- Must be keyword-dense (NOT sentence-style)\n"
            "- Prefer short noun phrases\n"
            "- Use comma or space separated terms only\n"
            "- Include synonyms only if they add new retrieval signals\n"
            "\n"
            "EXPANSION STRATEGY:\n"
            "- Add semantic synonyms: funding, grant, research funding, EU funding, call for proposals\n"
            "- Map concepts into EU research terminology (Horizon Europe, ERC, academic research)\n"
            "- Expand domain scope (academic, innovation, scientific research)\n"
            "- Include eligibility/funding context only as keywords (no sentences)\n"
            "\n"
            "GOAL:\n"
            "Maximize embedding recall and chunk overlap relevance in vector search."
        )
    },
    {
        "role": "user",
        "content": f"""
Original query:
{query}

Expanded query:
"""
    }
]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id)

    result = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True)

    return result.strip()