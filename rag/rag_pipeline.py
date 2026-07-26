from chroma_db import get_db
from .query_expansion import llm_expand_query, llm_extract_profile
import requests
#from llm import get_llm

retriever = None

def get_retriever():
    global retriever

    if retriever is None:
        db = get_db()
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20})

    return retriever

def rag_answer(query: str, pretty_print: bool = True ) -> dict:
    """
    Generate an answer using the grant RAG pipeline.

    The pipeline consists of four stages:

        1. Query Expansion
        2. Retrieval
        3. Context Construction
        4. Answer Generation

    If no relevant documents are found, the system returns
    an abstention response.

    Args:
        query (str):
            User question about grants, funding opportunities,
            deadlines, eligibility, or related topics.

        pretty_print (bool):
            Whether to print a formatted answer to the console.

    Returns:
        dict:
            Dictionary containing:

            - answer (str): Generated response.
            - sources (list[str]): Source URLs.
    """
    retriever = get_retriever()
    # 0. EXPAND
    expanded_query = llm_expand_query(query)

    # 1. RETRIEVE
    #results = db.similarity_search(expanded_query, k=5)
    results = retriever.invoke(expanded_query)

    grouped = {}

    for doc in results:
        doc_id = doc.metadata.get("id")
        text = doc.page_content

        if doc_id not in grouped:
            # create new doc
            grouped[doc_id] = doc
        else:
            # write to existing
            grouped[doc_id].page_content += "\n\n" + text[text.find("Fields:") + len("Fields:"):]

    results = list(grouped.values())

    if not results:
        return {"answer": "Insufficient evidence.", "sources": []}

    # 2. PACK CONTEXT
    context_chunks = []
    sources = []

    for i in range(len(results)):
        text = results[i].page_content
        context_chunks.append(f"[{i+1}]  {text}")
        context_chunks.append("start_date: " + str(results[i].metadata.get("start_date")))
        context_chunks.append("end_date: " + str(results[i].metadata.get("end_date")))
        context_chunks.append("min_amount: " + str(results[i].metadata.get("min_amount")))
        context_chunks.append("max_amount: " + str(results[i].metadata.get("max_amount")))
        if results[i].metadata.get("source"):
            sources.append(results[i].metadata.get("url"))

    context = "\n\n".join(context_chunks)

    # 3. PROMPT
    prompt = f"""
You are a RAG assistant for grant-related questions.

RULES:
- Use ONLY provided context. No external knowledge.
- Do NOT hallucinate or guess missing information.
- EVERY FACTUAL CLAIM must be supported by context and cited [source].
- If information is missing or insufficient, say:
  "Insufficient evidence"

TASKS:

1. Retrieval / Extraction:
Extract relevant grant info strictly from context with citations.

2. Paraphrased queries:
Understand intent but answer ONLY from retrieved data.

3. Structured listing:
Return:
- Grant name
- Short description
- Funding
- Deadline
- [source] citarion
If missing → "Not specified in sources"

4. Filtering:
Strictly filter by attributes using only context.

5. Comparison:
Use tables. No missing-data guessing.

6. Source verification:
Only confirm if explicitly supported by text.

7. Multi-turn:
Respect prior context and constraints.

8. Abstention:
If unclear or missing data → use strict refusal:
"Insufficient evidence"

ANTI-HALLUCINATION:
Never invent grants, numbers, deadlines, or eligibility rules.

STYLE:
Concise, structured, factual. No speculation.

ROLE:
You are a retrieval-based grant intelligence system, not a general knowledge model.

EVIDENCE:
{context}

QUESTION:
{query}

Answer:
"""

    #print(prompt)

    # 4. GENERATE
    response = requests.post(
        "http://127.0.0.1:8000/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False
        }
    )

    response.raise_for_status()

    gen = response.json()["answer"]

    if pretty_print:
        return pretty_print_rag({"answer": gen, "sources": list(sources)})
    else:
        return {"answer": gen, "sources": list(sources)}

def rag_cv_match(cv: str, pretty_print: bool = True ) -> dict:
    """
    Generate an answer using the grant RAG pipeline.
    """
    retriever = get_retriever()
    # 0. EXPAND
    profile = llm_extract_profile(cv)
    
    # 1. RETRIEVE
    results = retriever.invoke(profile)

    grouped = {}

    for doc in results:
        doc_id = doc.metadata.get("id")
        text = doc.page_content

        if doc_id not in grouped:
            # create new doc
            grouped[doc_id] = doc
        else:
            # write to existing
            grouped[doc_id].page_content += "\n\n" + text[text.find("Fields:") + len("Fields:"):]

    results = list(grouped.values())

    if not results:
        return {"answer": "Insufficient evidence.", "sources": []}

    # 2. PACK CONTEXT
    context_chunks = []
    sources = []

    for i in range(len(results)):
        text = results[i].page_content
        context_chunks.append(f"[{i+1}]  {text}")
        context_chunks.append("start_date: " + str(results[i].metadata.get("start_date")))
        context_chunks.append("end_date: " + str(results[i].metadata.get("end_date")))
        context_chunks.append("min_amount: " + str(results[i].metadata.get("min_amount")))
        context_chunks.append("max_amount: " + str(results[i].metadata.get("max_amount")))
        if results[i].metadata.get("source"):
            sources.append(results[i].metadata.get("url"))

    context = "\n\n".join(context_chunks)

    # 3. PROMPT
    prompt = f"""
You are an AI grant recommendation assistant.

TASK:

A researcher uploaded a CV.

A profile was extracted from the CV.

Your task is to recommend the most relevant grants from the retrieved context.

IMPORTANT:
- Use ONLY provided context.
- Do NOT invent grants.
- Do NOT invent funding values.
- Do NOT invent deadlines.
- Do NOT recommend grants not present in context.
- Explain WHY each grant matches the profile.
- If information is missing, write:
  "Not specified in sources"

RESEARCHER PROFILE:

{profile}

AVAILABLE GRANTS:

{context}

OUTPUT FORMAT:

1. **Grant Name**
   - Match: short explanation
   - Funding: ...
   - Deadline: ...
   - Source: [1]

2. **Grant Name**
   - Match: short explanation
   - Funding: ...
   - Deadline: ...
   - Source: [2]

Rank grants from most relevant to least relevant.
"""

    #print(prompt)

    # 4. GENERATE
    response = requests.post(
        "http://127.0.0.1:8000/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.0,
            "do_sample": False
        }
    )

    response.raise_for_status()

    gen = response.json()["answer"]

    if pretty_print:
        return pretty_print_rag({"answer": gen, "sources": list(sources)})
    else:
        return {"answer": gen, "sources": list(sources)}

def pretty_print_rag(result):
    """
    Format a RAG response for console output.

    Args:
        result: Dictionary returned by rag_answer().

    Returns:
        Formatted string.
    """

    lines = ["ANSWER:\n"]
    lines.append(result["answer"])

    lines.append("\nSOURCES:")

    for source in result.get("sources", []):
        lines.append(f"- {source}")

    if result.get("abstain"):
        lines.append("\nAbstained: no sufficient evidence")

    return "\n".join(lines)