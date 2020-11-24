# -*- coding: utf-8 -*-
import os
import sqlite3,json
from tqdm import tqdm
import unicodedata
def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)
def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with open(filename,encoding='utf-8') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            # Add the document
            documents.append((normalize(doc['id']),doc['text']))
            # print(doc['id'],normalize(doc['id']))
    return documents
conn = sqlite3.connect(r'DB_output/output.db')
os.chdir('data')
c = conn.cursor()
try:
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")
except:
    pass


for file in tqdm(os.listdir()):
    doc = get_contents(file)
    c.executemany("INSERT INTO documents VALUES (?,?)", doc)
conn.commit()
conn.close()