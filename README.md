# Grant Finder

A Retrieval-Augmented Generation (RAG) system for searching and recommending EU research grants.

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Collection

Collect grant data:

```bash
python collection/collect.py
```

This creates:

```text
data/processed/eu_grants_norm.jsonl
```

### Dataset Representation

The system uses a structured JSONL dataset where each entry represents a single EU funding opportunity.

Each grant is stored as a JSON object containing both metadata and a rich textual field used for embedding-based retrieval. The structure includes a unique identifier, source attribution, status information, temporal constraints (start and end dates), funding range, and a reference URL to the official EU funding portal.

A key component of the representation is the `embedding_text` field, which aggregates multiple descriptive elements such as the title, summary, keywords, thematic domains, expected outcomes, and full call description. This field is used as the primary input for embedding generation and semantic retrieval.

### Example Entry

```json
{
  "id": "HORIZON-CL5-2027-05-D4-07",
  "source": "eu",
  "status": "active",
  "start_date": "2027-05-05T00:00:00.000+0000",
  "end_date": "2027-09-15T00:00:00.000+0000",
  "min_amount": 5250000,
  "max_amount": 9400000,
  "url": "https://ec.europa.eu/info/funding-tenders/opportunities/portal/screen/opportunities/topic-details/HORIZON-CL5-2027-05-D4-07",
  "embedding_text": "Title: Integrating circularity in LCA-based modelling frameworks for ...
  \nSummary: Integrating circularity in LCA-based modelling frameworks for ...
  \nKeywords: HORIZON-CL5-2027-05-D4-07, HORIZON-CL5-2027-05
  \nFields: ..."
}
```

---

## Build Vector Database

Generate embeddings and populate ChromaDB:

```python
from chroma_db import ingest
from rag import llm_summary

ingest(llm_summary)
```

This creates the local Chroma database in:

```text
./chroma_db
```

---

## Run Web Interface

Start Flask:

```bash
python frontend/app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Mode 1 — Grant Search

Enter a natural-language query.

Example:

```text
AI grants for healthcare
```

Pipeline:

```text
Query
  ↓
Query Expansion
  ↓
Vector Retrieval
  ↓
Qwen Generation
  ↓
Answer + Sources
```

The system returns:

- relevant grants
- funding information
- deadlines
- source links

---

## Mode 2 — CV Matching

Upload a PDF CV.

Pipeline:

```text
PDF CV
  ↓
Text Extraction
  ↓
Research Profile Extraction
  ↓
Vector Retrieval
  ↓
Grant Recommendation
  ↓
Answer + Sources
```

The system:

1. extracts skills, research domains, and experience from the CV;
2. retrieves the most relevant grant opportunities;
3. explains why each grant matches the applicant profile;
4. returns supporting sources.