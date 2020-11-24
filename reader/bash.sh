#单独测试
torch-model-archiver --model-name reader --version 1.0 \
--serialized-file ./checkpoint_score_f1-86.233_em-66.853.pth \
--handler ./Transformer_handler_generalized.py \
--extra-files "./setup_config.json,./inference.py,./official_tokenization.py,./output.db,./entity.txt"
mkdir model_store
mv reader.mar model_store/
torchserve --start --ts-config config.properties --model-store model_store --models reader=reader.mar


mkdir model_store
mv NER/model_store/NER.mar model_store/
mv W2V/model_store/W2V.mar model_store/
cp W2V/config.properties config.properties
mv reader/model_store/reader.mar model_store/

torchserve --start --ts-config config.properties --model-store model_store --models NER=NER.mar,W2V=W2V.mar
torchserve --start --ts-config config.properties --model-store model_store --models reader=reader.mar,NER=NER.mar,W2V=W2V.mar





#test
import requests
import json
question = '习总书记毕业于哪里什么时候出生的?'.encode('utf-8')
a = requests.post('http://127.0.0.1:8080/predictions/reader',data=question).text
json.loads(a)
