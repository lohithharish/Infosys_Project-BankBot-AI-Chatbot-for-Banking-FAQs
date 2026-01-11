import sqlite3

conn = sqlite3.connect("bankbot.db")
cursor = conn.cursor()

# Add ₹500 to account 98765
cursor.execute(
    "UPDATE accounts SET balance = balance + ? WHERE account_number = ?",
    (100, "1")
)

conn.commit()
conn.close()

print("₹100 added to account 1")
