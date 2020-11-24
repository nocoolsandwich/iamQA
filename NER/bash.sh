torch-model-archiver --model-name NER --version 1.0 --serialized-file ./Transformer_handler_generalized.py --handler ./Transformer_handler_generalized.py --extra-files "./model/find_NER.py,./model/best_ner.bin,./model/SIM_main.py,./model/CRF_Model.py,./model/BERT_CRF.py,./model/NER_main.py"
mkdir model_store
mv NER.mar model_store/
torchserve --start --ts-config config.properties --model-store model_store --models NER=NER.mar

#mv NER/model_store/NER.mar W2V/model_store/NER.mar
torchserve --start --ts-config config.properties --model-store model_store --models NER=NER.mar W2V=W2V.mar

#test
import requests
question = '美国特勤局特工通常佩戴什么装备?'.encode('utf-8')
requests.post('http://127.0.0.1:8080/predictions/NER',data=question).text
