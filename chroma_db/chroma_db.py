from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import re
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

PATH = "data/processed/eu_grants_norm.jsonl"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

_db = None

def get_db():
    """
    Get or create the ChromaDB vector database instance.

    The database is initialized only once and reused across the application.
    Embeddings are generated using the all-MiniLM-L6-v2 sentence transformer.

    Returns:
        Chroma: Initialized ChromaDB collection containing grant documents.
    """
    global _db
    if _db is None:
        print("Loading db...")
        _db = Chroma(collection_name="grants", embedding_function=embeddings, persist_directory="./chroma_db")
    return _db

def ingest(llm_sum_func = None):
    """
    Load grant data from a JSONL file and populate the ChromaDB collection.

    For each grant:
        - extracts title, summary, keywords, and fields
        - removes code-like keywords
        - builds a compact document header
        - optionally compresses long headers using the LLM
        - splits grant fields into chunks
        - attaches metadata
        - stores chunks in ChromaDB

    Each chunk consists of max 1000 symbols of text + max 750 symbols header. Additionally collects chunk length statistics and visualizes them.

    Returns:
        None
    """
    if not llm_sum_func: 
        print("Please, pass llm_summary function as an argument")
        return
    stats = defaultdict(list)
    with open(PATH, "r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading grants"):
            item = json.loads(line)
            text = item.get("embedding_text", "")
            # cut header
            idx_summary = text.find("Summary:")
            idx_keywords = text.find("Keywords:")
            idx_domains = text.find("Domains:")
            idx_fields = text.find("Fields:")

            title = text[len("Title:"):idx_summary]
            summary = text[idx_summary + len("Summary:\n")+3:idx_keywords]
            #print("T:", title)
            #print("S:", summary)
            # filter kw
            keywords = text[idx_keywords + len("Keywords:"):idx_domains].strip()
            keywords_list = [k.strip() for k in keywords.split(",")]
            filtered_keywords = [
                k for k in keywords_list
                if not is_code_like(k)]

            keywords = ", ".join(filtered_keywords)
            # build header
            header = build_grant_header(title, summary, keywords)
            # llm summary if needed
            if len(header) > 750:
                header = llm_summary(summary)
            #print(header)
            
            # cut chunks
            body = text[idx_fields + len("Fields:"):].strip()
            body_chunks = splitter.split_text(body)
            chunks = [
                header + "\n\nFields: " + chunk#
                for chunk in body_chunks
            ]
            
            # metadata
            metadata = {
                "id": item.get('id'),
                "source": item.get("source"),
                "status": item.get("status"),
                "start_date": item.get("start_date"),
                "end_date": item.get("end_date"),
                "min_amount": item.get("min_amount"),
                "max_amount": item.get("max_amount"),
                "url": item.get("url")
            }

            # metadata for every chunk
            metadatas = [metadata for _ in chunks]
            # unique ids for chunks
            ids = [
                f"{item.get('id')}_chunk_{i}"
                for i in range(len(chunks))
            ]
            # collect stats
            for i, ch in zip(ids, chunks):
                doc_id = int(i.rsplit("_", 1)[1])
                stats[doc_id].append(len(ch))

            # add chunks to chroma
            _db.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=ids
            )

    print(len(stats),len(stats[0]),len(stats[1]),len(stats[2]))
    plot_chunk_len(stats)
    print("Done.")

def is_code_like(text):
    """
    Check whether a keyword resembles a technical identifier or grant code.

    Examples:
        HORIZON-2024
        ERC12345678

    Args:
        text (str):
            Keyword candidate.

    Returns:
        bool:
            True if the text matches the code pattern,
            otherwise False.
    """
    return bool(re.fullmatch(r"[A-Z0-9\-]{8,}", text))

def build_grant_header(title, summary, keywords):
    """
    Build a retrieval-oriented grant header.

    The header combines the grant title, summary, and optionally
    filtered keywords into a compact textual representation.

    Args:
        title (str):
            Grant title.

        summary (str):
            Grant summary.

        keywords (str):
            Comma-separated keyword list.

    Returns:
        str:
            Formatted grant header.
    """
    header = f"""Title: {title} \nSummary: {summary}"""

    if keywords:
       header += f"\nKeywords: {keywords}"

    return header

def plot_chunk_len(data):
    """
    Visualize chunk length distribution for each chunk position.

    Generates a boxplot showing how chunk sizes vary across
    the dataset after document splitting.

    Args:
        data (dict):
            Dictionary where keys represent chunk indices and
            values contain lists of chunk lengths.

    Returns:
        None
    """
    plt.boxplot(list(data.values()), labels=list(data.keys()),
        flierprops=dict(
        marker='.',
        markersize=3,
        markerfacecolor='red',
        markeredgecolor='red'))

    plt.xticks(rotation=45)
    plt.title("Chunk length per document")
    plt.ylabel("length")
    plt.show()