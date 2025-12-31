import sqlite3, os
os.makedirs("feedback", exist_ok=True)

def init_db():
    conn = sqlite3.connect("feedback/feedback.db")
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback(
        id INTEGER PRIMARY KEY,
        image TEXT,
        predicted INTEGER,
        correct INTEGER
    )
    """)
    conn.commit()
    conn.close()

def save(image, predicted, correct):
    conn = sqlite3.connect("feedback/feedback.db")
    c = conn.cursor()
    c.execute("INSERT INTO feedback VALUES(NULL,?,?,?)",
              (image, predicted, correct))
    conn.commit()
    conn.close()