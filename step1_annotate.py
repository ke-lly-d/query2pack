import os, json, time, threading, enlighten, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from star_utils import query_llm_proxy, get_llm_proxy_api_token

MAX_WORKERS     = 50              
REQS_PER_MIN    = 300             
MAX_RETRIES     = 3
BASE_SLEEP_SEC  = 1               
BACKEND         = "openai_direct_chat_gpt4o"
API_TOKEN       = get_llm_proxy_api_token()
INPUT_PATH      = "user_tasks.json"
OUTPUT_PATH     = "task_tool_mapping_out.json"

# json-schema and prompt ───────────────────────────────────
task_tool_mapping_item = {
    "type": "object",
    "properties": {
        "tool_to_search": {"type": "string"}
    },
    "required": ["tool_to_search"],
    "additionalProperties": False
}

schema = {
    "name": "task_tool_mapping",
    "strict": True,
    "type": "object",
    "properties": {
        "task_tool_mappings": {
            "type": "array",
            "items": task_tool_mapping_item,
            "minItems": 1
        }
    },
    "required": ["task_tool_mappings"],
    "additionalProperties": False
}

PROMPT = r"""
You are an employee who must complete **10 distinct tasks**.
Before doing each task you will SEARCH for supporting resources
(templates, SOPs, past examples, documentation).

**Your job:**  
For **each task**, decide which **one** internal tool you would click FIRST
to find those resources.

────────────────────────────────────────────────────────
ROLE CONTEXT
  • Position: {employee_tenure} {job_title}  
  • Department: {department}  
  • Company:   {company_size}-size {company_sector} organisation
────────────────────────────────────────────────────────
ALLOWED TOOLS — choose from *this list only*:  
  Coda · Confluence · Jira · Google Doc · Slack · Miro ·  
  Google Slides · Google Sheets · Gmail · GitHub
────────────────────────────────────────────────────────
OUTPUT RULES  
  • Pick **exactly one** tool *from the list above* for every task.  
  • Do **NOT** invent tools or return multiple tools.  
  • Return **only** valid JSON that matches the schema below—no prose.

    Example:
    {{
      "task_tool_mappings": [
        {{ "tool_to_search": "Confluence" }},
        ...
      ]
    }}

TASKS  
{formatted_tasks}
""".strip()


def build_prompt(row):
    tasks = "\n".join(f"{i}. {t}" for i, t in enumerate(row["tasks"]))
    return PROMPT.format(
        employee_tenure=row["employee_tenure"],
        job_title=row["job_title"],
        department=row["department"],
        company_size=row["company_size"],
        company_sector=row["company_sector"],
        formatted_tasks=tasks
    )

# load existing rows (if any) to avoid duplicates ───────────────────────────────────
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH) as f:
        existing_rows = json.load(f).get("task_tool_mappings", [])
else:
    existing_rows = []

seen = {r["user_idx"] for r in existing_rows}
seen_lock = threading.Lock()                 # thread-safe access

# global rate gate ───────────────────────────────────
_interval = 60 / REQS_PER_MIN
_last, gate_lock = time.time() - _interval, threading.Lock()

def gate():
    global _last
    with gate_lock:
        wait = _interval - (time.time() - _last)
        if wait > 0: time.sleep(wait)
        _last = time.time()

# worker with duplicate-check ───────────────────────────────────
def worker(row_dict):
    key = row_dict["user_idx"]

    # skip if already processed (another thread or previous run)
    with seen_lock:
        if key in seen:
            return []
        seen.add(key)

    prompt = build_prompt(row_dict)

    for att in range(MAX_RETRIES):
        try:
            gate()
            resp = query_llm_proxy(
                prompt,
                backend=BACKEND,
                schema=schema,
                sleep_time=0,
                api_token=API_TOKEN
            )
            txt = resp.get("chunk", {}).get("text", "{}")
            rows = json.loads(txt)["task_tool_mappings"]
            fixed = [
                {
                    "user_idx": row_dict["user_idx"],
                    "task_idx": i,
                    "task": row_dict["tasks"][i],
                    "tool_to_search": tool_row["tool_to_search"]
                }
                for i, tool_row in enumerate(rows)
            ]
            return fixed
        except Exception:
            if att == MAX_RETRIES - 1:
                return []
            time.sleep(BASE_SLEEP_SEC * 2**att)

# run pool, extend list, write once ───────────────────────────────────
def main():
    df = pd.read_json(INPUT_PATH, lines=True)
    results = existing_rows[:]            # start with what we already had

    mgr  = enlighten.get_manager()
    pbar = mgr.counter(total=len(df), desc="Annotating", unit="rows")

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futs = [pool.submit(worker, r.to_dict()) for _, r in df.iterrows()]
        for fut in as_completed(futs):
            results.extend(fut.result())  
            pbar.update()

    mgr.stop()

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"task_tool_mappings": results}, f, indent=2)

    print("✅ finished — unique rows:", len(results))

if __name__ == "__main__":
    main()