# main_app.py

import streamlit as st
import json
import re
from pathlib import Path
from typing import Dict, List
from db_files.db_setup import log_query



from db_files.db_setup import init_db

init_db()


# ---------- Paths & page config ----------
PROJECT_ROOT = Path(__file__).parent.resolve()
INTENTS_PATH = PROJECT_ROOT / "data" / "intents.json"

MODEL_META_PATH = PROJECT_ROOT / "nlu_engine" / "model_meta.json"
INTENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="BankBot NLU", layout="wide")

st.title("Lohith harish's BankBot NLU Visualizer")





# ---------- Helpers: load / save intents ----------
def load_intents() -> Dict[str, Dict]:
    """Robust loader that accepts multiple shapes and returns mapping intent->{"examples": [...] }."""
    if not INTENTS_PATH.exists():
        return {}
    raw = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
    # Case: {"intents": {...}}
    if isinstance(raw, dict) and "intents" in raw and isinstance(raw["intents"], dict):
        return raw["intents"]
    # Case: {"intents": [ {name, examples}, ... ]}
    if isinstance(raw, dict) and "intents" in raw and isinstance(raw["intents"], list):
        out = {}
        for it in raw["intents"]:
            if isinstance(it, dict):
                name = it.get("name") or it.get("intent")
                if name:
                    out[name] = {"examples": it.get("examples", [])}
        return out
    # Case: top-level list
    if isinstance(raw, list):
        out = {}
        for it in raw:
            if isinstance(it, dict):
                name = it.get("name") or it.get("intent")
                if name:
                    out[name] = {"examples": it.get("examples", [])}
        return out
    # Case: maybe mapping of intent->obj
    if isinstance(raw, dict):
        return raw
    return {}


def save_intents(intents: Dict[str, Dict]):
    INTENTS_PATH.write_text(
        json.dumps({"intents": intents}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------- Model meta helpers (for Train model section) ----------
def load_model_meta() -> Dict:
    if not MODEL_META_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def train_dummy_model(
    intents: Dict[str, Dict], epochs: int, batch_size: int, learning_rate: float
) -> Dict:
   

    meta = {
        "n_intents": len(intents),
        "total_examples": sum(len(v.get("examples", [])) for v in intents.values()),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        
    }
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


# ---------- NLU utilities ----------
def demo_score_intents(query: str, intents: Dict[str, Dict]) -> List[Dict]:
    

    q = (query or "").lower()
    q_tokens = re.findall(r"[0-9A-Za-z]+", q)
    results = []
    for name, info in intents.items():
        examples = info.get("examples", [])
        best = 0.0
        for ex in examples:
            ex_tokens = re.findall(r"[0-9A-Za-z]+", (ex or "").lower())
            if not ex_tokens:
                continue
            # proportion of example tokens that appear in query
            common = sum(1 for t in ex_tokens if t in q_tokens)
            frac = common / max(1, len(ex_tokens))
            best = max(best, frac)
        name_tokens = re.findall(r"[0-9A-Za-z]+", name.lower())
        if any(t in q_tokens for t in name_tokens):
            best = max(best, 0.6)
        results.append({"intent": name, "score": float(best)})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def demo_extract_entities(query: str) -> List[Dict]:
    
    q = query or ""
    ents = []
    # amount patterns: Rs 500, 500 rs, $50, ₹1,000 etc.
    m = re.search(
        r"(?:(?:rs|inr|rupees|₹|\$)\s*([0-9\.,]+))|([0-9\.,]+)\s*(?:rs|inr|rupees|₹|\$)",
        q,
        re.I,
    )
    if m:
        val = m.group(1) or m.group(2)
        if val:
            ents.append({"entity": "amount", "value": val.replace(",", "")})
    # account number after words account/acct/a/c
    m2 = re.search(
        r"(?:account|acct|a/c|to account)\s*(?:no|number|#|:)?\s*([0-9]{4,20})", q, re.I
    )
    if m2:
        ents.append({"entity": "account_number", "value": m2.group(1)})
    # txn/utr/ref patterns
    m3 = re.search(
        r"(?:utr|txn(?:\s*id)?|transaction id|ref(?:erence)? no|ref)\s*[:\s-]*([A-Za-z0-9\-_/]+)",
        q,
        re.I,
    )
    if m3:
        ents.append({"entity": "transaction_id", "value": m3.group(1)})
    return ents











# ---------- Load intents & model meta ----------
intents = load_intents()
model_meta = load_model_meta()


def handle_intent(intent, entities=None):
    if intent == "check_balance":
        return "💰 Your current account balance is ₹12,450.75"

    elif intent == "transfer_money":
        return "✅ Transfer initiated successfully."

    elif intent == "greeting":
        return "👋 Hello! How can I help you today?"

    elif intent == "thanks":
        return "😊 You're welcome!"

    else:
        return "⚠️ I can assist only with banking-related requests."

# ---------- Page layout: left / right ----------
left_col, right_col = st.columns([1.05, 1.5])










# ---------- Intents editor & Create new ----------
with left_col:
    st.header("Intents (edit & add)")

    if intents:
        # iterate copy of keys for safe mutate (rename/delete)
        for intent_key in list(intents.keys()):
            intent_info = intents[intent_key]
            examples = intent_info.get("examples", [])
            with st.expander(f"{intent_key} ({len(examples)} examples)", expanded=False):
                new_name = st.text_input(
                    "Rename intent", value=intent_key, key=f"name_{intent_key}"
                )
                examples_text = st.text_area(
                    "Examples (one per line)",
                    value="\n".join(examples),
                    height=150,
                    key=f"examples_{intent_key}",
                )
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Save", key=f"save_{intent_key}"):
                        updated_examples = [
                            e.strip() for e in examples_text.splitlines() if e.strip()
                        ]
                        if new_name and new_name != intent_key:
                            intents.pop(intent_key, None)
                            intents[new_name] = {"examples": updated_examples}
                        else:
                            intents[intent_key]["examples"] = updated_examples
                        save_intents(intents)
                        st.success("Saved changes.")
                with c2:
                    if st.button("Delete", key=f"del_{intent_key}"):
                        intents.pop(intent_key, None)
                        save_intents(intents)
                        st.experimental_rerun()
    else:
        st.info("No intents found. Create a new intent below.")

    st.markdown("---")
    st.header("Create new intent")
    new_intent_name = st.text_input("Intent name", key="new_intent_name")
    new_intent_examples = st.text_area(
        "Examples (one per line)", key="new_intent_examples", height=140
    )
    if st.button("Create Intent"):
        if not new_intent_name.strip():
            st.error("Provide an intent name.")
        else:
            intents[new_intent_name] = {
                "examples": [
                    e.strip() for e in new_intent_examples.splitlines() if e.strip()
                ]
            }
            save_intents(intents)
            st.success(f"Intent '{new_intent_name}' created. Refresh to view.")










# ---------- NLU Visualizer + Train model ----------
with right_col:
    st.header("NLU Visualizer")

    
    user_query = st.text_input(
        "User Query:",
        value="I want to transfer RS 500 from my savings account",
        key="user_query",
        max_chars=400,
    )

    st.write("Top intents to show")
    top_k_str = st.text_input("", value="4", key="top_k_textinput")
    try:
        top_k = int(top_k_str.strip())
        if top_k < 1:
            top_k = 1
    except Exception:
        top_k = 4

    st.write("")  
    if st.button("Analyze"):
        if not user_query.strip():
            st.error("Please enter a user query.")
        else:
            # Intent Recognition area
            st.subheader("Intent Recognition")
            preds = demo_score_intents(user_query, intents)
            if not preds:
                st.info("No intents available.")
            else:
                # show two-column list: left label, right score badge
                for p in preds[:top_k]:
                    intent_label = p["intent"].replace("_", " ").title()
                    score = p["score"]
                    a, b = st.columns([4, 1])
                    with a:
                        st.write(f"**{intent_label}**")
                    with b:
                        # blue-ish badge look using monospace text in a markdown/code style
                        st.write(f"`{score:.2f}`")

            st.markdown("---")
            # Entity Extraction area (green boxes)
            st.subheader("Entity Extraction")
            entities = demo_extract_entities(user_query)
            if not entities:
                st.write("No entities detected.")
            else:
                for ent in entities:
                    ent_name = ent.get("entity", "entity").replace("_", " ").title()
                    ent_value = ent.get("value")
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        # show small green box with icon
                        if ent_name.lower().startswith("amount"):
                            st.success(f"💰 {ent_name}")
                        else:
                            st.success(f"🔎 {ent_name}")
                    with c2:
                        st.info(f"Value: `{ent_value}`")

                        # ---------- Training Data Info ----------
st.markdown("---")
st.header("Training Data")

total_intents = len(intents)
total_examples = sum(len(v.get("examples", [])) for v in intents.values())

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Total Intents", total_intents)
with c2:
    st.metric("Total Examples", total_examples)
with c3:
    avg = round(total_examples / total_intents, 2) if total_intents else 0
    st.metric("Avg Examples / Intent", avg)












    # -------- Train model section --------
    st.markdown("---")
    st.header("Train model")


    epochs = st.number_input(
        "Epochs",
        min_value=1,
        max_value=200,
        value=10,
        step=1,
        key="epochs_input",
    )

    batch_size = st.number_input(
        "Batch size",
        min_value=1,
        max_value=256,
        value=8,
        step=1,
        key="batch_size_input",
    )

    learning_rate = st.number_input(
        "Learning rate",
        min_value=1e-6,
        max_value=1.0,
        value=0.01,
        step=0.0001,
        format="%.6f",
        key="lr_input",
    )

    if st.button("Start Training"):
        if not intents:
            st.error("No intents to train.")
        elif total_examples == 0:
            st.error("No training examples found.")
        else:
            with st.spinner("Training model..."):
                meta = train_dummy_model(
                    intents=intents,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )

            st.success("Model trained successfully!")
            st.subheader("Training Summary")
            st.json(meta)


            
