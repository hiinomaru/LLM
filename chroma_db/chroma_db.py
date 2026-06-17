from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

db = Chroma(collection_name="grants", embedding_function=embeddings, persist_directory="./chroma_db")

def ingest():
    stats = defaultdict(list)
    with open("data/processed/eu_grants_norm.jsonl", "r", encoding="utf-8") as file:

        for line in tqdm(file, desc="Loading grants"):
            item = json.loads(line)
            text = item.get("embedding_text", "")
            # first chunk
            idx = text.find("Fields:")
            if idx != -1:
                chunks = [text[:idx]]
                chunks.extend(splitter.split_text(text[idx:]))
            else:
                chunks = splitter.split_text(text)
            # metadata
            metadata = {
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
            db.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=ids
            )
    print(len(stats),len(stats[0]),len(stats[1]),len(stats[2]))
    plot_chunk_len(stats)
    print("Done.")

def plot_chunk_len(data):

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

ingest()