# -*- coding: utf-8 -*-
# stop = open('stop.txt',encoding='utf-8').read().split()
# print(len(stop))
# print(stop[0:10])
import pickle,sqlite3
import os,pandas as pd
from tqdm import tqdm
db_path = r'C:\Users\nocoo\Desktop\CMRC问答系统\reader\output.db'
conn = sqlite3.connect(db_path)
doc_ids = pd.read_sql_query("select id from documents",con=conn)['id'].tolist()
titles =  [i.split('|||')[0] for i in doc_ids]
with open('W2V', 'rb') as f:
    W2V = pickle.load(f)


need = []
for title in tqdm(titles):
    try:
        need+=[i[0] for i in W2V.most_similar(title, topn=5)]+[title]
    except:
        print(title)
print(len(need))
with open(r"sgns.merge.word",encoding='utf-8') as f:
    for index,line in enumerate(f):
        word = line.split(' ')[0]
        if word in need:
            with open(r'imited_w2v.txt','a',encoding='utf-8') as w:
                print(word)
                w.write(line)
        # if index==3:
        #     break