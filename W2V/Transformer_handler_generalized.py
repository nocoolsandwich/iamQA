# -*- coding: utf-8 -*-
from abc import ABC
import logging
import os
import gensim
import pickle
from ts.torch_handler.base_handler import BaseHandler

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
        serialized_file = self.manifest['model']['serializedFile']

        model_pt_path = os.path.join(model_dir, serialized_file)
#         with open(model_pt_path, 'rb') as f:
#             self.W2V = pickle.load(f)
        self.W2V = gensim.models.KeyedVectors.load_word2vec_format(model_pt_path,binary = False)



        #Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        #further setup config can be added.

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
        output = []
        for idx, data in enumerate(requests):
            text = data.get("data")
            if text is None:
                text = data.get("body")
            input_text = text.decode('utf-8')

            ################input处理
            question = input_text
            try:
                words = self.W2V.most_similar(question,topn=5)
                words = [i[0] for i in words]
            except:
                print(f'{question}不在W2V词典')
                words = ['']
            print('your entity:{}\nclose entity:{}'.format(question,words))
            output.append(words)
            ################处理完毕
        return output

    def inference(self, input_batch):
        """ Predict the class (or classes) of the received text using the serialized transformers checkpoint.
        """
        inferences = []
        # Handling inference for token_classification.
        batch_size = len(input_batch)

        num_rows = batch_size
        for i in range(num_rows):
            inferences.append({'close entity':input_batch})
        logger.info("Model predicted: '%s'", input_batch)

        return inferences

    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        return inference_output
