import streamlit as st
import sqlite3
import re
import pandas as pd
from datetime import datetime
from llama_cpp import Llama
from nlu_engine.infer_intent import infer_intent
# =========================
# INTERNAL IMPORTS
# =========================

from db_files.db_setup import init_db, log_query
from db_files.auth import login_user, register_user


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🏦 BankBot AI",
    layout="wide"
)

# =========================
# INIT DB
# =========================
init_db()

# =========================
# SESSION STATE
# =========================
if "user" not in st.session_state:
    st.session_state.user = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_tx" not in st.session_state:
    st.session_state.pending_tx = None

if "show_password" not in st.session_state:
    st.session_state.show_password = False


# =========================
# LOAD LLM
# =========================
@st.cache_resource
def load_llama():
    return Llama(
        model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_ctx=1024,          # 🔥 smaller context
        n_threads=8,
        n_gpu_layers=0,
        verbose=False
    )


llama = None


# =========================
# DB CONNECTION
# =========================
def get_db():
    return sqlite3.connect("bankbot.db", check_same_thread=False)

conn = get_db()
cursor = conn.cursor()

# =========================
# LLM RESPONSE
# =========================
def get_llm_response(text):
    global llama
    if llama is None:
        llama = load_llama()

    prompt = f"""
<|system|>
You are a helpful assistant. Answer clearly.
<|user|>
{text}
<|assistant|>
"""

    output = llama(
    prompt,
    max_tokens=80,      # 🔥 limit tokens
    temperature=0.2,
    stop=["<|user|>"]
)

    response = output["choices"][0]["text"]

    # CLEAN RESPONSE
    response = response.replace("<|assistant|>", "")
    response = response.replace("<|user|>", "")
    response = response.replace("<|system|>", "")

    return response.strip()


    



# =========================
# BANKING HANDLER
# =========================
def handle_banking(text, account_number):
    text = text.lower()
    # =========================
    # PASSWORD ENTERED IN CHAT
    # =========================
    if st.session_state.show_password and st.session_state.pending_tx:
        user = login_user(st.session_state.user["username"], text)

        if not user:
            st.session_state.pending_tx = None
            st.session_state.show_password = False
            return "❌ Incorrect password. Transaction cancelled."

        tx = st.session_state.pending_tx

        sender = cursor.execute(
            "SELECT balance FROM accounts WHERE account_number=?",
            (tx["from"],)
        ).fetchone()

        receiver = cursor.execute(
            "SELECT balance FROM accounts WHERE account_number=?",
            (tx["to"],)
        ).fetchone()

        if not sender or sender[0] < tx["amount"]:
            return "❌ Insufficient balance"
        if not receiver:
            return "❌ Receiver not found"

        cursor.execute(
            "UPDATE accounts SET balance = balance - ? WHERE account_number=?",
            (tx["amount"], tx["from"])
        )
        cursor.execute(
            "UPDATE accounts SET balance = balance + ? WHERE account_number=?",
            (tx["amount"], tx["to"])
        )
        cursor.execute("""
            INSERT INTO transactions (from_account, to_account, amount, timestamp)
            VALUES (?, ?, ?, ?)
        """, (tx["from"], tx["to"], tx["amount"], datetime.now()))

        conn.commit()

        st.session_state.pending_tx = None
        st.session_state.show_password = False

        return f"✅ ₹{tx['amount']} sent successfully"


    # BALANCE
    if "balance" in text:
        

        bal = cursor.execute(
            "SELECT balance FROM accounts WHERE account_number=?",
            (account_number,)
        ).fetchone()
        return f"💰 Your balance is ₹{bal[0]}" if bal else "❌ Account not found"

    # DEPOSIT
    if "deposit" in text or "add" in text:
        amount = float(re.search(r"\d+", text).group())

        cursor.execute(
            "UPDATE accounts SET balance = balance + ? WHERE account_number=?",
            (amount, account_number)
        )
        cursor.execute("""
            INSERT INTO transactions (from_account, to_account, amount, timestamp)
            VALUES (?, ?, ?, ?)
        """, ("CASH", account_number, amount, datetime.now()))

        conn.commit()
        return f"✅ ₹{amount} deposited successfully"


    # SEND MONEY
    # SEND MONEY (STEP 1: ASK PASSWORD)
    match = re.search(r"(send|transfer)\s+(\d+)\s+to\s+(\d+)", text)
    if match:
        amount = float(match.group(2))
        to_acc = match.group(3)

        # Store transaction temporarily
        st.session_state.pending_tx = {
            "from": account_number,
            "to": to_acc,
            "amount": amount
        }
        st.session_state.show_password = True

        return "🔐 Please enter your password to confirm the transaction"
    



    return None

# =========================
# LOGIN / REGISTER
# =========================
if not st.session_state.user:
    st.title("🔐 Login / Register")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login_user(u, p)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        ru = st.text_input("New Username")
        rp = st.text_input("New Password", type="password")
        acc = st.text_input("Account Number")
        if st.button("Register"):
            success = register_user(ru, rp, acc)

            if success:
                st.success("✅ Account created successfully. Please login.")
            else:
                st.error("❌ Account number already exists. Try a different one.")


    st.stop()

# =========================
# USER DETAILS
# =========================
user = st.session_state.user
account_number = user["account_number"]


# =========================
# SIDEBAR
# =========================
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home", "💬 Chatbot", "📊 Transactions", "ℹ️ Help", "🚪 Logout"]
)

# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.title("🏦 BankBot AI Dashboard")

    # ---------------------------
    # CURRENT BALANCE
    # ---------------------------
    bal = cursor.execute(
        "SELECT balance FROM accounts WHERE account_number=?",
        (account_number,)
    ).fetchone()

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Current Balance", f"₹ {bal[0] if bal else 0}")

    # ---------------------------
    # TOTAL CREDIT & DEBIT
    # ---------------------------
    credit = cursor.execute("""
        SELECT COALESCE(SUM(amount),0) FROM transactions
        WHERE to_account=?
    """, (account_number,)).fetchone()[0]

    debit = cursor.execute("""
        SELECT COALESCE(SUM(amount),0) FROM transactions
        WHERE from_account=?
    """, (account_number,)).fetchone()[0]

    col2.metric("📥 Total Credit", f"₹ {credit}")
    col3.metric("📤 Total Debit", f"₹ {debit}")

    st.divider()

    # ---------------------------
    # INCOME VS EXPENSE GRAPH
    # ---------------------------
    st.subheader("📊 Income vs Expense")

    chart_df = pd.DataFrame({
        "Type": ["Credit", "Debit"],
        "Amount": [credit, debit]
    })

    st.bar_chart(chart_df.set_index("Type"))

    st.divider()

    # ---------------------------
    # TRANSACTION TREND
    # ---------------------------
    st.subheader("📈 Transaction Trend")

    tx_df = pd.read_sql("""
        SELECT timestamp, amount FROM transactions
        WHERE from_account=? OR to_account=?
        ORDER BY timestamp
    """, conn, params=(account_number, account_number))

    if not tx_df.empty:
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])
        st.line_chart(tx_df.set_index("timestamp")["amount"])
    else:
        st.info("No transactions yet")

    st.divider()

    # ---------------------------
    # RECENT TRANSACTIONS
    # ---------------------------
    st.subheader("🧾 Recent Transactions")

    recent = cursor.execute("""
        SELECT from_account, to_account, amount, timestamp
        FROM transactions
        WHERE from_account=? OR to_account=?
        ORDER BY timestamp DESC
        LIMIT 5
    """, (account_number, account_number)).fetchall()

    if recent:
        df_recent = pd.DataFrame(
            recent,
            columns=["From", "To", "Amount", "Time"]
        )
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No recent transactions")


# =========================
# CHATBOT
# =========================
elif page == "💬 Chatbot":
    st.title("💬 BankBot Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask your banking query...")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        result = infer_intent(user_msg)
        intent = result["intent"]
        confidence = result["confidence"]

        BANKING_INTENTS = [
            "check_balance",
            "deposit_money",
            "transfer_money",
            "send_money"
        ]

        SMALL_TALK_INTENTS = [
            "greeting",
            "help"
        ]

        # 🔐 PASSWORD FLOW HAS PRIORITY
        if st.session_state.show_password:
            reply = handle_banking(user_msg, account_number)

        elif intent in BANKING_INTENTS and confidence >= 0.6:
            reply = handle_banking(user_msg, account_number)

        elif intent in SMALL_TALK_INTENTS:
            reply = "👋 Hi! How can I help you with your bank account today?"

        else:
            # 🚀 Skip LLM for very short messages
            if len(user_msg.strip()) <= 3:
                reply = "🙂 Please ask a complete question."
            else:
                with st.spinner("🤖 Thinking..."):
                    reply = get_llm_response(user_msg)



        log_query(user_msg, intent, confidence, int(confidence > 0.7))

        st.session_state.messages.append(
            {"role": "assistant", "content": reply}
        )

        st.rerun()


# =========================
# TRANSACTIONS
# =========================
elif page == "📊 Transactions":
    st.title("📜 Transaction Analytics")

    # ---------------------------
    # LOAD TRANSACTIONS
    # ---------------------------
    df = pd.read_sql("""
        SELECT id, from_account, to_account, amount, timestamp
        FROM transactions
        WHERE from_account=? OR to_account=?
        ORDER BY timestamp DESC
    """, conn, params=(account_number, account_number))

    if df.empty:
        st.info("No transactions found.")
        st.stop()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------------------------
    # CREDIT / DEBIT COLUMN
    # ---------------------------
    df["type"] = df.apply(
        lambda x: "Credit" if x["to_account"] == account_number else "Debit",
        axis=1
    )

    # ---------------------------
    # SUMMARY METRICS
    # ---------------------------
    credit = df[df["type"] == "Credit"]["amount"].sum()
    debit = df[df["type"] == "Debit"]["amount"].sum()
    net = credit - debit

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📥 Total Credit", f"₹ {credit}")
    col2.metric("📤 Total Debit", f"₹ {debit}")
    col3.metric("📊 Net Change", f"₹ {net}")
    col4.metric("🧾 Transactions", len(df))

    st.divider()

    # ---------------------------
    # INCOME VS EXPENSE BAR CHART
    # ---------------------------
    st.subheader("📊 Income vs Expense")

    bar_df = pd.DataFrame({
        "Type": ["Credit", "Debit"],
        "Amount": [credit, debit]
    })

    st.bar_chart(bar_df.set_index("Type"))

    st.divider()

    # ---------------------------
    # TRANSACTION TREND
    # ---------------------------
    st.subheader("📈 Transaction Trend")

    trend_df = df.sort_values("timestamp")
    st.line_chart(trend_df.set_index("timestamp")["amount"])

    st.divider()

    # ---------------------------
    # CREDIT VS DEBIT PIE CHART
    # ---------------------------
    st.subheader("🥧 Credit vs Debit")

    pie_df = df.groupby("type")["amount"].sum()
    st.pyplot(pie_df.plot.pie(autopct="%1.1f%%").figure)

    st.divider()

    # ---------------------------
    # FILTERS
    # ---------------------------
    st.subheader("🔍 Filter Transactions")

    tx_type = st.selectbox("Transaction Type", ["All", "Credit", "Debit"])

    if tx_type != "All":
        df = df[df["type"] == tx_type]

    # ---------------------------
    # TRANSACTIONS TABLE
    # ---------------------------
    st.subheader("🧾 Transaction Details")

    st.dataframe(
        df[["from_account", "to_account", "type", "amount", "timestamp"]],
        use_container_width=True
    )

    st.divider()

    # ---------------------------
    # DOWNLOAD OPTIONS
    # ---------------------------
    st.subheader("📥 Download Data")

    csv = df.to_csv(index=False)
    json_data = df.to_json(orient="records", indent=2)

    col1, col2 = st.columns(2)

    col1.download_button(
        "⬇️ Download CSV",
        csv,
        file_name="transactions.csv",
        mime="text/csv"
    )

    col2.download_button(
        "⬇️ Download JSON",
        json_data,
        file_name="transactions.json",
        mime="application/json"
    )


# =========================
# HELP
# =========================
elif page == "ℹ️ Help":
    st.title("❓ BankBot Help & User Guide")

    st.markdown("""
    Welcome to **BankBot AI** 🤖  
    This assistant helps you manage your bank account using **natural language commands**.
    """)

    st.divider()

    # ---------------------------
    # WHAT CAN BANKBOT DO
    # ---------------------------
    st.subheader("🧭 What can BankBot do?")

    st.markdown("""
    ✅ Check account balance  
    ✅ Deposit money  
    ✅ Transfer money securely (with password confirmation)  
    ✅ View transaction history & analytics  
    ✅ Download transactions as CSV or JSON  
    """)

    st.divider()

    # ---------------------------
    # EXAMPLE COMMANDS
    # ---------------------------
    st.subheader("💬 Example Commands")

    st.markdown("""
    **Balance**
    ```
    check my balance
    show my account balance
    ```

    **Deposit**
    ```
    deposit 500
    add 1000 to my account
    ```

    **Transfer**
    ```
    send 200 to 12345
    transfer 50 to 2
    ```
    """)

    st.divider()

    # ---------------------------
    # PASSWORD CONFIRMATION
    # ---------------------------
    st.subheader("🔐 Secure Transfer (Password Confirmation)")

    st.markdown("""
    For security, **every money transfer requires password confirmation**.

    **How it works:**
    1. You request a transfer  
    2. BankBot asks for your password  
    3. You type your password **in the same chat box**  
    4. Transfer completes only if password is correct  

    **Example:**
    ```
    send 100 to 2
    → Please enter your password
    1
    → ₹100 sent successfully
    ```
    """)

    st.info("🔒 Your password is never displayed or stored in chat history.")

    st.divider()

    # ---------------------------
    # DASHBOARD INFO
    # ---------------------------
    st.subheader("📊 Home Dashboard Insights")

    st.markdown("""
    On the **Home page**, you can see:
    - Current balance  
    - Total money credited & debited  
    - Income vs Expense graph  
    - Transaction trend over time  
    - Recent transactions  
    """)

    st.divider()

    # ---------------------------
    # TRANSACTIONS PAGE
    # ---------------------------
    st.subheader("📜 Transactions & Analytics")

    st.markdown("""
    The **Transactions page** provides:
    - Income vs Expense chart  
    - Credit vs Debit pie chart  
    - Transaction trend graph  
    - Filters by transaction type  
    - Download options (CSV / JSON)  
    """)

    st.divider()

    # ---------------------------
    # DOWNLOAD HELP
    # ---------------------------
    st.subheader("📥 Downloading Data")

    st.markdown("""
    You can download your transaction data for:
    - Reports  
    - Excel analysis  
    - Record keeping  

    **Available formats:**
    - CSV (Excel-friendly)
    - JSON (Developer-friendly)
    """)

    st.divider()

    # ---------------------------
    # LIMITATIONS
    # ---------------------------
    st.subheader("⚠️ Limitations")

    st.markdown("""
    - BankBot supports **basic banking commands only**
    - LLM responses are for **general knowledge**, not financial advice
    - Password must be typed **exactly**
    """)

    st.divider()

    # ---------------------------
    # DEMO TIPS
    # ---------------------------
    st.subheader("🎤 Demo Tips (For Presentation)")

    st.markdown("""
    🔹 Start with balance check  
    🔹 Show a secure transfer with password  
    🔹 Navigate to Transactions → show graphs  
    🔹 Download CSV file  
    🔹 Explain security flow  
    """)

    st.success("✅ You are ready to use BankBot confidently!")


# =========================
# LOGOUT
# =========================
elif page == "🚪 Logout":
    st.session_state.clear()
    st.success("Logged out successfully")
    st.rerun()
