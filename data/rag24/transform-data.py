#!/usr/bin/env python3
from hashlib import md5
from pathlib import Path
import json
import gzip
try:
    from tqdm import tqdm
except ImportError:  # tqdm ships in the [all] extra; fall back to a no-op wrapper
    def tqdm(iterable, *args, **kwargs):
        return iterable
import ir_datasets

EXPECTED_MD5 = "918bb96e714eec2e3a0c64d0771a6a6f"


def read_all_lines():
    txt = Path("nugget_assignment.20241108.jl.jl").read_text()
    actual_md5 = md5(txt.encode("UTF-8")).hexdigest()
    assert actual_md5 == EXPECTED_MD5
    for l in txt.split("\n"):
        if not l:
            continue
        yield json.loads(l)


def extract_topics():
    qid_to_topic = {}
    for topic in read_all_lines():
        qid_to_topic[topic["qid"]] = topic["query"]
    
    Path("topics").mkdir(parents=True, exist_ok=True)
    with open("topics/topics_rag24.jsonl", "w") as f:
        for qid, title in qid_to_topic.items():
            f.write(json.dumps({"request_id": qid, "title": title}) + "\n")


def extract_responses():
    Path("runs/generation/").mkdir(parents=True, exist_ok=True)
    run_ids = set()
    topic_ids = set()
    run_id_to_responses = {}
    ds = ir_datasets.load("msmarco-segment-v2.1").docs_store()

    for response in read_all_lines():
        run_ids.add(response["run_id"])
        topic_ids.add(response["qid"])

    for run_id in run_ids:
        run_id = run_id.split(".")[0]
        run_id = run_id.replace("manual-manual", "manual")
        with gzip.open(f"raw-responses/{run_id}.gz", "rt") as f:
            for l in f:
                try:
                    l = json.loads(l)
                    if l["topic_id"] not in topic_ids:
                        continue
                    if run_id not in run_id_to_responses:
                        run_id_to_responses[run_id] = []
                    l["run_id"] = run_id
                    
                    l["metadata"] = {"team_id": run_id, "run_id": run_id, "topic_id": l["topic_id"], "request_id": l["topic_id"], "narrative_id": l["topic_id"]}
                    
                    run_id_to_responses[run_id].append(l)
                except:
                    pass

    for run_id in run_id_to_responses.keys():
        assert len(run_id_to_responses[run_id]) > 19, run_id
        with open(f"runs/generation/{run_id}.jsonl", "w") as f:
            for response in tqdm(run_id_to_responses[run_id]):
                docs = {}
                for r in response["references"]:
                    doc = ds.get(r)
                    docs[r] = {"id": r, "text": doc.default_text(), "title": doc.title}
                response["documents"] = docs
                f.write(json.dumps(response) + "\n")


if __name__ == '__main__':
    extract_topics()
    extract_responses()
