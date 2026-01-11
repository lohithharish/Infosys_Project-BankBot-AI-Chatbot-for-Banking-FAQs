import json
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = DATA_DIR / "chat_history.json"

def log_chat(user, query, response):
    history = []
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())

    history.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "query": query,
        "response": response
    })

    HISTORY_FILE.write_text(json.dumps(history, indent=4))

def history_ui():
    st.markdown("## 📜 Chat History 🧾")

    if not HISTORY_FILE.exists():
        st.info("✨ No conversations yet")
        return

    data = json.loads(HISTORY_FILE.read_text())
    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "⬇️ Download JSON",
        json.dumps(data, indent=4),
        "chat_history.json",
        "application/json"
    )

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False),
        "chat_history.csv",
        "text/csv"
    )
