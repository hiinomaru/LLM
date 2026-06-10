import meilisearch
import json
from tqdm import tqdm


def filter_project(proj):
    # сразу отсекаем мусор
    if not (
        (proj.get("_str.status.en") == "starting date pending") and
        proj.get("_str.projecttitle.en") and
        proj.get("_str.prproposalsummary.en")
    ):
        return None

    result = {
        "source": "fwf",
        "id": proj.get("id"),
        "title": proj.get("_str.projecttitle.en"),
        "summary": proj.get("_str.prproposalsummary.en"),
        "status": proj.get("_str.status.en"),
        "start_date": proj.get("_date.startdate"),
        "end_date": proj.get("_date.enddate"),
        "keywords": proj.get("_list.keywords.split", []),
        "domains": proj.get("_list.researchareas.en", []),
        "fields": proj.get("_list.researchfields.en", []),
        "amount": proj.get("_long.approvedamount"),
        "url": proj.get("_str.url"),
    }

    return result

client = meilisearch.Client(
    "https://openapi.fwf.ac.at",
    "3a03f2f39cc8a99ea0775270adb4946c425469aa7f291e7ca9f2d8424337c1af"
)

def get_fwf_grants(filename="fwf_grants.jsonl"):
    offset = 0
    limit = 1000

    with open(filename, "w", encoding="utf-8") as f:
        with tqdm(desc="Fetching + filtering") as pbar:

            while True:
                batch = client.index('projects').get_documents({'limit': limit, 'offset': offset})
                if not batch.results:
                    break

                for proj in batch.results:
                    proj = filter_project(dict(proj))
                    if proj:
                        f.write(json.dumps(proj, ensure_ascii=False) + "\n")

                offset += limit
                pbar.update(len(batch.results))

    print("FWF - done")