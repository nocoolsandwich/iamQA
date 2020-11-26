# wikiCH_DRrQA
中文wiki百科问答系统，本项目使用了torchserver部署模型    

知识库:wiki百科中文数据 

模型:使用了的NER(CCKS2016数据)和阅读理解模型(CMRC2018)，还有W2V词向量搜索。  
# 模块介绍
- ChineseWiki-master
功能：清洗wiki中文数据
相关项目:https://github.com/mattzheng/ChineseWiki  
- NER
功能：从问题中识别实体  
例子：qurry:周董是谁？  》》 entiy:周董  
模型：ALBERT
数据集：CCKS2016KBQA  
相关项目：https://github.com/997261095/bert-kbqa
- Word2vec
功能：如果实体不在知识库，则用W2V搜索近似实体  
例子：entity:周董 >> ['周杰伦','JAY','林俊杰']  
相关项目：https://github.com/Embedding/Chinese-Word-Vectors
- Entity linking
功能:根据NER或W2V得到的mention entity搜索知识库  
- Reader
功能：阅读理解文段，精确定位答案。  
例子：参考SQuQA  
模型：ALBERT  
数据集：CMRC2018  
相关项目：https://github.com/CLUEbenchmark/CLUE
- Web
功能：web服务，前端交互和结果呈现  
相关项目：https://github.com/zaghaghi/drqa-webui
# 使用说明
1. 安装torchserve

    参考[install-torchserve](https://github.com/pytorch/serve#install-torchserve)   
    
2. 安装requirements.txt

    使用豆瓣源快些  
    
    ```bash
    pip install -U -r requirements.txt -i https://pypi.douban.com/simple/
    ```
    
3. 下载准备文件  

- wiki中文数据,[下载地址](https://dumps.wikimedia.org/zhwiki/) 

    linux可用    
    
    ```bash
    wget https://dumps.wikimedia.org/zhwiki/20201120/zhwiki-20201120-pages-articles-multistream.xml.bz2
    ```
    
    文件大小约2G,需要放入ChineseWiki-master根目录
    
- NER的albert模型  

    模型我已训练好，文件总大小约16M，下载地址   
    
    drive:[NER](https://drive.google.com/file/d/14HWqT9LDuF9kvbKFI95TziiHSI9O2BL-/view?usp=sharing)
    
    baiduyun：  
    
    下载后存放路径:`NER\model`  
    
- reader的albert模型  

    模型我已训练好，文件总大小约35M，下载地址   
    
    drive:[reader](https://drive.google.com/file/d/1rQnT4j95oHkEbS5oQi6ecLkuhjzM0lRO/view?usp=sharing)
    
    baiduyun:
    
    下载后存放路径:`reader`  
    
- W2V
    [下载地址](https://github.com/Embedding/Chinese-Word-Vectors)  
    
    Word2vec/Skip-Gram with Negative Sampling (SGNS)下的Mixed-large 综合Baidu Netdisk/Google Drive的Word  
    
    或者通过这其中一个链接下载:[drive](https://drive.google.com/open?id=1Zh9ZCEu8_eSQ-qkYVQufQDNKPC4mtEKR)[百度](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)
    
    下载解压后将`sgns.merge.word`存放路径:`W2V`   

4. wiki数据清洗    

    依次运行`1wiki_to_txt.py`,`2wiki_txt_to_csv.py`,`3wiki_csv_to_json.py`,`4wiki_json_to_DB.py`  
    
    输出:`ChineseWiki-master\DB_output\output.db`,然后把`output.db`放入reader下  

5. torchserve打包模型,启动服务  

    在`NER`目录执行  
    
    ```bash
    torch-model-archiver --model-name NER --version 1.0 \
    --serialized-file ./Transformer_handler_generalized.py \
    --handler ./Transformer_handler_generalized.py --extra-files \
    "./model/find_NER.py,./model/best_ner.bin,./model/SIM_main.py,./model/CRF_Model.py,./model/BERT_CRF.py,./model/NER_main.py"
    ```
    
    在`reader`目录执行  
    
    ```bash
    torch-model-archiver --model-name reader --version 1.0 \
    --serialized-file ./checkpoint_score_f1-86.233_em-66.853.pth \
    --handler ./Transformer_handler_generalized.py \
    --extra-files "./setup_config.json,./inference.py,./official_tokenization.py,./output.db"
    ``` 
    在`W2V`目录执行   
    
    ```bash
    torch-model-archiver --model-name W2V --version 1.0 --serialized-file ./W2V --handler ./Transformer_handler_generalized.py
    ```  
    
    在`wikiCH_QA`目录执行
    
    ```bash
    mkdir model_store \
    mv NER/NER.mar model_store/ \
    mv W2V/W2V.mar model_store/ \
    cp W2V/config.properties config.properties \
    mv reader/reader.mar model_store/ \
    ```  
    
    启动服务
    
    ```bash
    torchserve --start --ts-config config.properties --model-store model_store \
    --models reader=reader.mar,NER=NER.mar,W2V=W2V.mar
    ```  
    
# 项目说明  
NER模块在CCKS2016KBQA准确率98%   
reader模块在CMRC2018EM:66%,F1:86%  
<img src='https://pic3.zhimg.com/80/v2-d878daa1f3d754c927319efd8dfe8e56_1440w.jpg' class='width:200px'></img>

    
   
    
 
