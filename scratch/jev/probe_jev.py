"""Ask Jev to pick among the same BM25 candidates Qwen saw; compare to Qwen's answer."""
import asyncio, csv, json, os, random, time, statistics
import httpx

URL = "https://openrouter.ai/api/alpha/decisions"
KEY = os.environ["XWALK_TEST_API_KEY"]
K = 25
random.seed(7)

source = {r["ID"]: r for r in csv.DictReader(open("data/ref_zivila/source.csv", encoding="utf-8-sig"))}
results = [json.loads(l) for l in open("runs/ref_zivila/foodon_qwen/results.jsonl")]
by_status = {}
for r in results:
    by_status.setdefault(r["status"], []).append(r)
sample = (random.sample(by_status["matched"], 30) + random.sample(by_status["needs_review"], 15)
          + random.sample(by_status["unmatched"], 15))
no_en = [r for r in results if not source[r["source_id"]]["mention_en"]]
sample += random.sample(no_en, 10)

def cand_text(c):
    f = c["record"]["fields"]
    parts = [f["label"]]
    if f.get("synonyms"): parts.append("synonyms: " + "; ".join(f["synonyms"]))
    if f.get("definition"): parts.append("definition: " + f["definition"][:200])
    if f.get("parent_labels"): parts.append("parents: " + "; ".join(f["parent_labels"]))
    return " | ".join(parts)

def build(r):
    s = source[r["source_id"]]
    src = {k: v for k, v in {"english_name": s["mention_en"], "slovenian_name": s["name_slo"],
           "slovenian_tag_name": s["tag_name_slo"], "slovenian_short_name": s["short_name_slo"],
           "food_group_slovenian": s["fgnm"], "curation_warning": s["curation_note"]}.items() if v}
    cands = r["candidates"][:K]
    keys = [f"C{i+1:02d}" for i in range(len(cands))]
    state = {"source_record": src, "candidates": {k: cand_text(c) for k, c in zip(keys, cands)}}
    crit = {k: cand_text(c)[:160] for k, c in zip(keys, cands)}
    crit["NONE"] = "no candidate denotes the same food as the source record"
    rules = ("Select the candidate that denotes the same food or food material as `source_record`. "
             "Processing state (raw, cooked, dried, canned, frozen, juice, oil, flour) and species or body part "
             "must agree when stated. A dish is not one of its ingredients. Never select a process, quality, "
             "specification, or organism merely because its label overlaps. If no candidate is the same food, choose NONE.")
    questions = {"best": {"type": "choice", "instructions": rules, "criteria": crit}}
    for k in keys:
        questions[f"same_{k}"] = {"type": "noul",
            "instructions": f"Does `candidates.{k}` denote the same food or food material as `source_record`, with processing state and species agreeing when stated?"}
    return {"model": "~typesafe/jev-latest", "state": state, "questions": questions}, keys, cands

async def one(client, sem, r):
    body, keys, cands = build(r)
    if not keys:
        return {"source_id": r["source_id"], "error": "no candidates"}
    async with sem:
        for attempt in range(4):
            t = time.time()
            resp = await client.post(URL, headers={"authorization": f"Bearer {KEY}"}, json=body)
            dt = time.time() - t
            if resp.status_code in (429, 529, 500, 502, 503):
                await asyncio.sleep(2 ** attempt); continue
            break
    if resp.status_code != 200:
        return {"source_id": r["source_id"], "error": f"{resp.status_code} {resp.text[:200]}"}
    a = resp.json()
    best = a["answers"]["best"]
    idmap = {k: c["record"]["id"] for k, c in zip(keys, cands)}
    labmap = {k: c["record"]["fields"]["label"] for k, c in zip(keys, cands)}
    nouls = {k: a["answers"][f"same_{k}"]["noul"] for k in keys}
    top_noul_key = max(nouls, key=nouls.get)
    return {"source_id": r["source_id"], "en": source[r["source_id"]]["mention_en"], "slo": source[r["source_id"]]["name_slo"],
            "qwen_status": r["status"], "qwen_id": r["matched_id"], "qwen_label": (r.get("matched_record") or {}).get("fields", {}).get("label"),
            "qwen_conf": r["confidence"],
            "jev_choice": best["choice"], "jev_id": idmap.get(best["choice"]), "jev_label": labmap.get(best["choice"]),
            "jev_conf": best["confidence"], "jev_p": best["probabilities"].get(best["choice"]),
            "jev_p_none": best["probabilities"].get("NONE"),
            "noul_top_id": idmap[top_noul_key], "noul_top": nouls[top_noul_key],
            "noul_of_qwen": next((nouls[k] for k in keys if idmap[k] == r["matched_id"]), None),
            "n_cands": len(keys),
            "qwen_in_top25": any(idmap[k] == r["matched_id"] for k in keys),
            "latency": dt, "tokens": a["usage"]["input_tokens"], "cost": a["usage"]["cost"], "model": a["model"]}

async def main():
    sem = asyncio.Semaphore(4)
    async with httpx.AsyncClient(timeout=60) as client:
        out = await asyncio.gather(*(one(client, sem, r) for r in sample))
    json.dump(out, open(os.environ["S"] + "/probe_results.json", "w"), indent=1, ensure_ascii=False)
    ok = [o for o in out if "error" not in o]
    print("calls", len(out), "errors", len(out) - len(ok))
    for o in out:
        if "error" in o: print("ERR", o)
    print("latency p50/p95 s", round(statistics.median(o["latency"] for o in ok), 2), round(sorted(o["latency"] for o in ok)[int(0.95 * len(ok))], 2))
    print("tokens/call mean", round(statistics.mean(o["tokens"] for o in ok)), "total cost $", round(sum(o["cost"] for o in ok), 5))
    for st in ["matched", "needs_review", "unmatched"]:
        g = [o for o in ok if o["qwen_status"] == st]
        agree = sum(1 for o in g if o["jev_id"] == o["qwen_id"])
        none_ = sum(1 for o in g if o["jev_choice"] == "NONE")
        print(f"qwen={st:13s} n={len(g):2d} jev_same_id={agree:2d} jev_NONE={none_:2d} mean_jev_conf={statistics.mean(o['jev_conf'] for o in g):.2f}")
    print("\n--- disagreements / interesting rows ---")
    for o in ok:
        flag = "" if o["jev_id"] == o["qwen_id"] else "DIFF"
        print(f"{flag:4s} [{o['qwen_status'][:7]}] {o['en'] or '-'} / {o['slo']!r:40.40} | qwen={o['qwen_label']!r} ({o['qwen_conf']}) | jev={o['jev_label']!r} p={o['jev_p']:.2f} pNONE={o['jev_p_none']:.2f} | noulTop={o['noul_top']:.2f} noulQwen={o['noul_of_qwen']} n={o['n_cands']}")

asyncio.run(main())
