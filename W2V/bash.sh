torch-model-archiver --model-name W2V --version 1.0 --serialized-file ./W2V --handler ./Transformer_handler_generalized.py
mkdir model_store
mv W2V.mar model_store/
torchserve --start --ts-config config.properties --model-store model_store --models W2V=W2V.mar
#test
import requests
question = '特朗普'.encode('utf-8')
requests.post('http://127.0.0.1:8080/predictions/W2V',data=question).text
