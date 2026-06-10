import requests
import json
from tqdm import tqdm
import re
from bs4 import BeautifulSoup


def unwrap(x):
    if isinstance(x, list):
        return x[0] if x else None
    return x

def filter_project(proj):
    metadata = proj.get("metadata", {})

    statuses = metadata.get("status", [])

    VALID_STATUSES = {"31094501", "31094502"}

    if not any(s in VALID_STATUSES for s in statuses):
        return None

    result = {
        "source": "eu",
        "id": unwrap(metadata.get("identifier")),
        "title": unwrap(metadata.get("title")),
        "summary": proj.get("summary") or proj.get("content"),
        "status": unwrap(statuses),
        "start_date": unwrap(metadata.get("startDate")),
        "end_date": unwrap(metadata.get("deadlineDate")),
        "keywords": metadata.get("keywords", []),
        "domains": metadata.get("focusArea", []),
        "fields": metadata.get("destinationDetails", []),
        "description": metadata.get("descriptionByte", []),
        "amount": unwrap(metadata.get("budgetOverview", [])),
        "url": proj.get("url"),
    }

    return result

def get_eu_grants(filename):
    url = "https://api.tech.ec.europa.eu/search-api/prod/rest/search"

    query_data = {
        "bool": {
            "must": [
                {
                    "terms": {
                        "status": ["31094501", "31094502"]
                    }
                }
            ]
        }
    }

    files = {
        "query": (None, json.dumps(query_data), "application/json"),

        "languages": (None, json.dumps(["en"]), "application/json"),

        "sort": (
            None,
            json.dumps({
                "field": "startDate",
                "order": "DESC"
            }),
            "application/json"
        )
    }

    page = 1
    page_size = 100

    params = {
        "apiKey": "SEDIA",
        "text": "*",
        "pageSize": page_size,
        "pageNumber": page}


    with open(filename, "w", encoding="utf-8") as f:
        with tqdm(desc="Fetching + filtering") as pbar:

            while True:
                params["pageNumber"] = page
                response = requests.post(
                    url,
                    params=params,
                    files=files)

                data = response.json()
                batch = data.get("results", [])
                if not batch:
                    break

                for proj in batch:
                    proj = filter_project(dict(proj))
                    if proj:
                        f.write(json.dumps(proj, ensure_ascii=False) + "\n")

                page += 1
                pbar.update(len(batch))
    
    print("EU - done")

#Cleaning and formatting#

def clean_html(text):
    if not text:
        return ""

    if isinstance(text, list):
        text = " ".join(text)

    if not isinstance(text, str):
        return str(text)

    return BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

EU_STATUS_MAP = {
    "31094501": "active",
    "31094502": "pending",
}

import json
import math

def extract_budget_min_max(budget):
    if not budget:
        return {
            "min_budget": None,
            "max_budget": None
        }

    if isinstance(budget, str):
        data = json.loads(budget)
    else:
        data = budget

    min_val = math.inf
    max_val = -math.inf
    found = False

    topic_map = data.get("budgetTopicActionMap", {})

    for actions in topic_map.values():
        for a in actions:
            if not isinstance(a, dict):
                continue

            min_c = a.get("minContribution")
            max_c = a.get("maxContribution")

            for v in (min_c, max_c):
                if isinstance(v, (int, float)):
                    min_val = min(min_val, v)
                    max_val = max(max_val, v)
                    found = True

                elif isinstance(v, str) and v.isdigit():
                    v = int(v)
                    min_val = min(min_val, v)
                    max_val = max(max_val, v)
                    found = True

    return {
        "min_budget": None if not found else min_val,
        "max_budget": None if not found else max_val
    }

def normalize_status(source, status):
    if not status:
        return None
    if isinstance(status, list):
        status = status[0]
    return EU_STATUS_MAP.get(status, status)

    return status

def normalize_record(item):
    source = item["source"]
    amount = extract_budget_min_max(item.get("amount"))

    embedding_text = f"""
    Title: {item.get("title") or ""}
    Summary:
    {item.get("summary") or ""}
    Keywords:
    {", ".join(item.get("keywords") or [])}
    Domains:
    {", ".join(item.get("domains") or [])}
    Fields:
    {clean_html(item.get("fields") or "")}
    Description:
    {clean_html(item.get("description") or "")}
    """.strip()

    return {
        "id": item.get("id"),
        "source": source,
        "status": normalize_status(source, item.get("status")),
        "start_date": item.get("start_date"),
        "end_date": item.get("end_date"),
        "min_amount": amount.get("min_budget"),
        "max_amount": amount.get("max_budget"),
        "url": item.get("url"),
        "embedding_text": embedding_text
    }

def process_eu_jsonl(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Processing JSONL"):
            item = json.loads(line)

            normalized = normalize_record(item)
            if not normalized:
                continue

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")