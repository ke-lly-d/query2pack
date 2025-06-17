import os, json, time, threading, pandas as pd, enlighten
from concurrent.futures import ThreadPoolExecutor, as_completed
from star_utils import query_llm_proxy, get_llm_proxy_api_token

BACKEND   = "openai_direct_o3"
TOKEN     = get_llm_proxy_api_token()
IN_FILE   = "annotation_1_out.json"    
OUT_FILE  = "validation_out.json"

MAX_WORKERS, REQS_PER_MIN = 50, 300   
MAX_RETRIES, BASE_SLEEP   = 3, 1

# json schema and prompt ───────────────────────────────────
validation_item = {
    "type": "object",
    "properties": {
        "ok":       {"type": "string", "enum": ["Y", "N"]},
        "better":   {"type": "string"}
    },
    "required": ["ok", "better"],
    "additionalProperties": False
}

schema = {
    "name": "tool_validation",
    "strict": True,
    "type": "object",
    "properties": {
        "tool_validation": {
            "type": "array",
            "items": validation_item,
            "minItems": 10,
            "maxItems": 10
        }
    },
    "required": ["tool_validation"],
    "additionalProperties": False
}

PROMPT = """
You will validate **10** task–tool annotations for ONE employee.

Allowed tools   (choose only from this list)
  • Coda
  • Confluence
  • Jira
  • Google Doc
  • Slack
  • Miro
  • Google Slides
  • Google Sheets
  • Gmail
  • GitHub

Goal
----
Decide whether the original tool is **clearly the single best place** an
employee would click FIRST to find helpful resources (templates, SOPs, past
examples, documentation) for each task.

Guidelines
----------
1. KEEP the original tool (“ok":"Y") **when it is a strong, obvious fit**.  
2. If the original tool is not in the list **or** another tool is *more*
   obviously useful, CHANGE it (“ok":"N") and give ONE better tool from the
   allowed list.  
3. Aim for the tool that would save the employee the *most time*.
4. For tasks that are primarily **real-time, external-facing, or
   networking-oriented** (e.g., attending a conference, reaching out to
   journalists, scheduling calls), prefer tools that store conversation
   history or contacts (Slack, Gmail).
5. Wiki repositories (Coda, Confluence, Google Doc) are preferred only when the
   task explicitly involves policies, templates, past written reports,
   or formal knowledge-base content.

Return ONLY JSON that matches this schema:

{{
  "tool_validation": [
    {{
      "ok": "Y" | "N",
      "better": "-" | "<one tool from the list>"
    }},
    ...
  ]
}}
─────────────────────────────────────────────────────────────
{bullet_block}
""".strip()

def bullet_block(rows):
    return "\n\n".join(
        f"#{r['task_idx']}\nTask : {r['task']}\nAnnot: {r['tool_to_search']}"
        for r in rows
    )

# global rate-gate ───────────────────────────────────
_interval = 60 / REQS_PER_MIN
_last, gate_lock = time.time() - _interval, threading.Lock()
def gate():
    global _last
    with gate_lock:
        wait = _interval - (time.time() - _last)
        if wait > 0: time.sleep(wait)
        _last = time.time()

# worker ───────────────────────────────────
def validate_user(user_rows):
    prompt = PROMPT.format(bullet_block=bullet_block(user_rows))
    for att in range(MAX_RETRIES):
        try:
            gate()
            resp = query_llm_proxy(
                prompt,
                backend=BACKEND,
                schema=schema,
                sleep_time=0,
                api_token=TOKEN
            )
            verdicts = json.loads(resp["chunk"]["text"])["tool_validation"]
            # merge back by order 0-9
            for i, row in enumerate(user_rows):
                row.update(verdicts[i])
            return user_rows
        except Exception:
            if att == MAX_RETRIES-1:
                for r in user_rows:
                    r.update({"ok":"ERR","better":"-"})
                return user_rows
            time.sleep(BASE_SLEEP*2**att)


# main ───────────────────────────────────
def main():
    rows = [json.loads(l) for l in open(IN_FILE)]
    df   = pd.DataFrame(rows)

    groups = [g.sort_values("task_idx").to_dict("records")
              for _, g in df.groupby("user_idx", sort=False)]

    mgr  = enlighten.get_manager()
    pbar = mgr.counter(total=len(groups), desc="Validating", unit="users")

    with open(OUT_FILE, "w") as fout, \
         ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = [pool.submit(validate_user, grp) for grp in groups]
        for fut in as_completed(futures):
            for row in fut.result():           # 10 lines per user
                fout.write(json.dumps(row) + "\n")
            pbar.update()
    mgr.stop()
    print("wrote", OUT_FILE)


if __name__ == "__main__":
    main()
