import csv
import sqlite3

# Step 1: Read the CSV file
data = []
with open('data/train.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append(row)

# Step 2: Create a SQLite database
conn = sqlite3.connect('data/train.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY,
    task TEXT,
    answer TEXT
)
''')

# Step 3: Insert data into the database
for row in data:
    cursor.execute('''
    INSERT INTO tasks (task, answer)
    VALUES (?, ?)
    ''', (row['task'], row['answer']))

# Commit the changes and close the connection
conn.commit()
conn.close()
