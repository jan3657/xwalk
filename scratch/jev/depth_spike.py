"""Spike: does Jev's accuracy hold as the candidate set grows? Measured against gold on the 200-target samples."""
import asyncio, csv, json, os, sys, time, statistics
import httpx
from xwalk.config import load_job
from xwalk.retrieval.bm25 import BM25Retriever
from xwalk.retrieval.base import SearchRequest
from xwalk.stores.memory import MemoryStore

URL = "https://openrouter.ai/api/alpha/decisions"
H = {"authorization": f"Bearer {os.environ['XWALK_TEST_API_KEY']}"}
EX = sys.argv[1]; S = os.environ["S"]
job = load_job(f"examples/{EX}/job.yaml"); T = job.build_templates()
store = MemoryStore.from_source(job.build_target_records()); all_ids = sorted(store.ids()) if hasattr(store, "ids") else None
bm25 = BM25Retriever.open(f"{S}/idx_{EX}/bm25")
sources = {r.id: r for r in job.build_source_records()}
gold = {r["source_id"]: frozenset(x for x in r["gold_ids"].split("|") if x) for r in csv.DictReader(open(f"examples/{EX}/sample/gold.csv"))}
slots = job.build_prompts().slots
RULES = (f"Select the candidate {slots.target_noun} that denotes the same entity as the {slots.entity_noun} in `source`. "
         + " ".join(slots.hard_rules) + " Choose NONE if no candidate denotes the same entity.")
NOUL = f"Does this candidate {slots.target_noun} denote the same entity as the {slots.entity_noun} in `source`? " + " ".join(slots.hard_rules[:2])

def ctext(rid):
    r = store.get(rid); f = r.fields
    parts = [str(f.get("label", ""))]
    if f.get("synonyms"): parts.append("synonyms: " + "; ".join(map(str, f["synonyms"]))[:200])
    if f.get("definition"): parts.append("definition: " + str(f["definition"])[:160])
    return " | ".join(parts)

async def retrieve(src, k):
    hits = await bm25.search(SearchRequest(text=T.render_query(src), limit=k, source_record=src))
    return [h.record_id for h in hits]

async def ask(client, sem, src, ids, with_choice):
    keys = [f"C{i+1:03d}" for i in range(len(ids))]
    src_state = {"mention": T.render_query(src)}
    ctx = T.render_context(src)
    if ctx and ctx != src_state["mention"]: src_state["context"] = ctx[:1200]
    state = {"source": src_state, "candidates": {k: ctext(i) for k, i in zip(keys, ids)}}
    qs = {f"n_{k}": {"type": "noul", "instructions": NOUL.replace("this candidate", f"`candidates.{k}`")} for k in keys}
    if with_choice:
        crit = {k: ctext(i)[:140] for k, i in zip(keys, ids)}; crit["NONE"] = "no candidate denotes the same entity"
        qs["best"] = {"type": "choice", "instructions": RULES, "criteria": crit}
    body = {"model": "~typesafe/jev-latest", "state": state, "questions": qs}
    async with sem:
        for attempt in range(5):
            t = time.time(); r = await client.post(URL, headers=H, json=body); dt = time.time() - t
            if r.status_code in (429, 500, 502, 503, 529): await asyncio.sleep(1.5 ** attempt); continue
            break
    if r.status_code != 200: return {"error": f"{r.status_code} {r.text[:150]}"}
    a = r.json()
    nouls = {i: a["answers"][f"n_{k}"]["noul"] for k, i in zip(keys, ids)}
    out = {"nouls": nouls, "latency": dt, "tokens": a["usage"]["input_tokens"], "cost": a["usage"]["cost"]}
    if with_choice:
        b = a["answers"]["best"]; out["choice"] = None if b["choice"] == "NONE" else dict(zip(keys, ids)).get(b["choice"])
        out["p_choice"] = b["probabilities"][b["choice"]]; out["conf"] = b["confidence"]
    return out

async def main():
    sem = asyncio.Semaphore(6)
    ids_all = sorted({r.id for r in job.build_target_records()})
    rows = [(sid, sources[sid]) for sid in gold if sid in sources]
    async with httpx.AsyncClient(timeout=90) as client:
        # retrieval ceiling
        ret = {sid: await retrieve(src, 200) for sid, src in rows}
        reach = sum(1 for sid,_ in rows if gold[sid] & set(ids_all)) / len(rows)
        print(f"gold reachable (in target set): {reach:.2f}")
        for k in (10, 25, 50, 100, 200):
            print(f"recall@{k}: {sum(1 for sid,_ in rows if gold[sid] & set(ret[sid][:k]))/len(rows):.2f}")
        variants = {
            "bm25_25": lambda sid: [ret[sid][:25]],
            "all200_one_call": lambda sid: [ids_all],
            "all200_chunks50": lambda sid: [ids_all[i:i+50] for i in range(0, 200, 50)],
        }
        summary = {}
        for name, fn in variants.items():
            async def run_one(sid, src):
                chunks = [c for c in fn(sid) if c]; with_choice = len(chunks) == 1
                if not chunks: return {"sid": sid, "noul_top": None, "noul_top_p": 0.0, "noul_gold": None, "choice": None, "p_choice": None, "latency": 0, "tokens": 0, "cost": 0}
                res = await asyncio.gather(*(ask(client, sem, src, c, with_choice) for c in chunks))
                if any("error" in r for r in res): return {"sid": sid, "error": [r.get("error") for r in res]}
                nouls = {}; [nouls.update(r["nouls"]) for r in res]
                top = max(nouls, key=nouls.get)
                return {"sid": sid, "noul_top": top, "noul_top_p": nouls[top], "noul_gold": max((nouls[g] for g in gold[sid] if g in nouls), default=None),
                        "choice": res[0].get("choice") if with_choice else None, "p_choice": res[0].get("p_choice"),
                        "latency": max(r["latency"] for r in res), "tokens": sum(r["tokens"] for r in res), "cost": sum(r["cost"] for r in res)}
            out = await asyncio.gather(*(run_one(sid, src) for sid, src in rows))
            ok = [o for o in out if "error" not in o]; n = len(ok)
            noul_acc = sum(1 for o in ok if o["noul_top"] in gold[o["sid"]]) / n
            line = {"n": n, "errors": len(out) - n, "noul_top1_acc": round(noul_acc, 2),
                    "median_latency": round(statistics.median(o["latency"] for o in ok), 2), "mean_tokens": round(statistics.mean(o["tokens"] for o in ok)),
                    "cost_total": round(sum(o["cost"] for o in ok), 4)}
            if ok[0]["choice"] is not None or any(o["choice"] for o in ok):
                line["choice_acc"] = round(sum(1 for o in ok if o["choice"] in gold[o["sid"]]) / n, 2)
                line["choice_NONE"] = sum(1 for o in ok if o["choice"] is None)
            # calibration-ish: accuracy when noul_top_p >= 0.8 vs below
            hi = [o for o in ok if o["noul_top_p"] >= 0.8]; lo = [o for o in ok if o["noul_top_p"] < 0.8]
            line["acc_if_noul>=.8"] = (round(sum(1 for o in hi if o["noul_top"] in gold[o["sid"]]) / len(hi), 2), len(hi)) if hi else None
            line["acc_if_noul<.8"] = (round(sum(1 for o in lo if o["noul_top"] in gold[o["sid"]]) / len(lo), 2), len(lo)) if lo else None
            summary[name] = line
            print(name, json.dumps(line)); sys.stdout.flush()
            if name == "all200_one_call":
                wrong = [o for o in ok if o["noul_top"] not in gold[o["sid"]]][:6]
                for o in wrong:
                    print("   WRONG", T.render_query(sources[o["sid"]])[:40], "->", ctext(o["noul_top"])[:50], f"p={o['noul_top_p']:.2f}", "| gold:", [ctext(g)[:40] for g in gold[o["sid"]] if g in ids_all][:1], f"p_gold={o['noul_gold']}")
    json.dump(summary, open(f"{S}/depth_{EX}.json", "w"), indent=1)

asyncio.run(main())
