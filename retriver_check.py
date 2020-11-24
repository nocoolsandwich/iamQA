import sqlite3,requests
import pandas as pd
db_path = r'reader\output.db'
conn = sqlite3.connect(db_path)
doc_ids = pd.read_sql_query("select id from documents",con=conn)['id'].tolist()
titles =  [i.split('|||')[0] for i in doc_ids]
# print([i for i in titles if '川普' in i])
import jieba
import json


def document_retriever(query):
    entity = json.loads(requests.post('http://127.0.0.1:8080/predictions/NER', data=query.encode('utf-8')).text)
    entity = entity['entity']
    print(f'识别实体:{entity}')
    if entity and entity in titles:
        print('实体在知识库中')
        doc_id = doc_ids[titles.index(entity)]
        sql = "select text from documents where id = '{}'".format(doc_id)
        doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
        return doc, entity
    if entity:
        print(f'实体不在知识库,使用W2V近似搜索:{entity}')
        word = json.loads(
            requests.post('http://127.0.0.1:8080/predictions/W2V', data=entity.encode('utf-8')).text)
        word = word['close entity'][0]
        print(f'近似搜索结果:{word}')
    if word:
        for i in word:
            if i and i in titles:
                print(f'近似搜索成功,{query},转换成{i}')
                doc_id = doc_ids[titles.index(i)]
                sql = "select text from documents where id = '{}'".format(doc_id)
                doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
                return doc, i
    print('NER和W2V失败,使用jieba')
    cut = jieba.cut(query)
    cut = [i for i in cut]
    q = [i for i in cut for j in titles if j == i]
    if q:
        print(f'结巴搜索成功:{query}>>{entity}>>{word}>>{q}')
        doc_id = doc_ids[titles.index(q[0])]
        sql = "select text from documents where id = '{}'".format(doc_id)
        doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
        return doc, q[0]
    print('无法识别')
    return '真的真的不知道呀', '我不知道呀'

# a = document_retriever('习总书记毕业于哪里')
# # a = document_retriever('习主席毕业于哪里')
# print(a)
while True:
    q = input('输入问题:')
    a = document_retriever(q)
    print(a)