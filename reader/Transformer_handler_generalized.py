# -*- coding: utf-8 -*-
from abc import ABC
import json
import logging
import os
import requests
import math
import torch
# from drqa import retriever
from transformers import AlbertForQuestionAnswering, BertTokenizer, AutoConfig,AutoModelForTokenClassification
from ts.torch_handler.base_handler import BaseHandler
import sqlite3
import pandas as pd

logger = logging.getLogger(__name__)

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """
    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        from sys import path
        path.append(model_dir)
        from inference import json2features
        from inference import evaluate
        self.json2features = json2features
        self.evaluate = evaluate
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #read configs for the mode, model_name, etc. from setup_config.json

        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                # self.setup_config = json.loads(setup_config_file.read())
                self.setup_config = {
                                      "model_name": "voidful/albert_chinese_base",
                                      "mode": "sequence_classification",
                                      "do_lower_case": "True",
                                      "num_labels": "2",
                                      "save_mode": "pretrained",
                                      "max_length": "150"
                                    }
        else:
            logger.warning('Missing the setup_config.json file.')



        import jieba
        #jieba.load_userdict(os.path.join(model_dir,r'entity.txt'))
        #
        # tfidf_path = os.path.join(model_dir, r'output-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz')
        # ranker = retriever.get_class('tfidf')(
        #     tfidf_path=tfidf_path)
        db_path = os.path.join(model_dir,r'output.db')
        conn = sqlite3.connect(db_path)
        self.doc_ids = pd.read_sql_query("select id from documents",con=conn)['id'].tolist()
        self.titles =  [i.split('|||')[0] for i in self.doc_ids]

        import json
        def document_retriever(query):
            entity = json.loads(requests.post('http://127.0.0.1:8080/predictions/NER', data=query.encode('utf-8')).text)
            entity = entity['entity']
            print(f'识别实体:{entity}')
            if entity and entity in self.titles:
                print('实体在知识库中')
                doc_id = self.doc_ids[self.titles.index(entity)]
                sql = "select text from documents where id = '{}'".format(doc_id)
                doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
                return doc, entity
            if entity:
                print(f'实体不在知识库,使用W2V近似搜索:{entity}')
                word = json.loads(
                    requests.post('http://127.0.0.1:8080/predictions/W2V', data=entity.encode('utf-8')).text)
                print(word)
                word = word['close entity'][0]
                print(f'近似搜索结果:{word}')
            if word:
                for i in word:
                    if i and i in self.titles:
                        print(f'近似搜索成功,{query},转换成{i}')
                        doc_id = self.doc_ids[self.titles.index(i)]
                        sql = "select text from documents where id = '{}'".format(doc_id)
                        doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
                        return doc, i
            print('NER和W2V失败,使用jieba')
            cut = jieba.cut(query)
            cut = [i for i in cut]
            q = [i for i in cut for j in self.titles if j == i]
            if q:
                print('结巴搜索成功')
                doc_id = self.doc_ids[self.titles.index(q[0])]
                sql = "select text from documents where id = '{}'".format(doc_id)
                doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
                return doc, q[0]
            print('无法识别')
            return '真的真的不知道呀', '我不知道呀'
            # else:
            #     doc_id = ranker.closest_docs(query, k=1)[0][0]
            #     sql = "select text from documents where id = '{}'".format(doc_id)
            #     doc = pd.read_sql_query(sql, con=conn)['text'].tolist()[0]
            #     return doc,doc_id.split('|||')[0]

        self.retriever = document_retriever

        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.
        bert_config = AutoConfig.from_pretrained(self.setup_config["model_name"])
        self.model = AlbertForQuestionAnswering.from_pretrained(model_pt_path,**{'config':bert_config})
        tokenizer_kwards = {'do_lower_case': False, 'max_len': 512}
        self.tokenizer = BertTokenizer.from_pretrained(self.setup_config["model_name"], **tokenizer_kwards)
        self.model.to(self.device)
        self.model.eval()

        logger.debug('Transformer model from path {0} loaded successfully'.format(model_dir))

        # Read the mapping file, index to object name
        # mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        # # Question answering does not need the index_to_name.json file.
        # if not self.setup_config["mode"]== "question_answering":
        #     if os.path.isfile(mapping_file_path):
        #         with open(mapping_file_path) as f:
        #             self.mapping = json.load(f)
        #     else:
        #         logger.warning('Missing the index_to_name.json file.')
        self.initialized = True

    def preprocess(self, requests):
        """ Basic text preprocessing, based on the user's chocie of application mode.
        """
        input_batch = None
        for idx, data in enumerate(requests):
            text = data.get("data")
            if text is None:
                text = data.get("body")
            input_text = text.decode('utf-8')

            ################input处理
            question = input_text
            context,title = self.retriever(question)
            print('your question:{}\ndocument retriever:{}'.format(question, context))
            data = [{'paragraphs': [{'id': 'DEV_0', 'context': context,
                                     'qas': [{'id': 'QUERY_0', 'question': question, 'answers': [{'text': 'haha'}]}]}]}]


            example, feature = self.json2features(data, self.tokenizer, is_training=False,
                                             repeat_limit=3, max_query_length=64,
                                             max_seq_length=512, doc_stride=128)
            ################处理完毕
        return example,feature,[context],[title]

    def inference(self, input_batch):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        inferences = []
        # Handling inference for token_classification.
        batch_size = len(input_batch[0])
        outputs = self.evaluate(self.model,batch_size,
                           input_batch[0],input_batch[1],self.device)[0]['QUERY_0']

        num_rows = batch_size
        context = input_batch[2]
        for i in range(num_rows):
            inferences.append({'title':input_batch[3][i],'context':context[i],'answer':outputs})
        logger.info("Model predicted: '%s'", outputs)

        return inferences

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output
