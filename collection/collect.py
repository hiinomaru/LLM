from get_eu import get_eu_grants, process_eu_jsonl

EU_RAW_FILE = r"data\raw\eu_grants.jsonl"
EU_PROCESSED_FILE = r"data\processed\eu_grants_norm.jsonl"

get_eu_grants(EU_RAW_FILE)
process_eu_jsonl(EU_RAW_FILE, EU_PROCESSED_FILE)