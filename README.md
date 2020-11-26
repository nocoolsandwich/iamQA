# wikiCH_DRrQA
中文wiki百科问答系统，本项目使用了torchserver部署模型  
知识库:wiki百科中文数据,  
模型:使用了的NER(CCKS2016数据)和阅读理解模型(CMRC2018)，还有W2V词向量搜索。  
# 模块介绍
## ChineseWiki-master
功能：清洗wiki中文数据
相关项目:https://github.com/mattzheng/ChineseWiki  
## NER
功能：从问题中识别实体  
例子：qurry:周董是谁？  》》 entiy:周董  
模型：ALBERT
数据集：CCKS2016KBQA  
相关项目：https://github.com/997261095/bert-kbqa
## Word2vec
功能：如果实体不在知识库，则用W2V搜索近似实体  
例子：entity:周董 >> ['周杰伦','JAY','林俊杰']  
相关项目：https://github.com/Embedding/Chinese-Word-Vectors
## Entity linking
功能:根据NER或W2V得到的mention entity搜索知识库  
## Reader
功能：阅读理解文段，精确定位答案。  
例子：参考SQuQA  
模型：ALBERT  
数据集：CMRC2018  
相关项目：https://github.com/CLUEbenchmark/CLUE
## Web
功能：web服务，前端交互和结果呈现  
相关项目：https://github.com/zaghaghi/drqa-webui
# 使用说明
1.安装torchserve
    参考[install-torchserve](https://github.com/pytorch/serve#install-torchserve)
2.安装requirements.txt
    ```bash
    pip install -U -r requirements.txt -i https://pypi.douban.com/simple/
    ```
3.下载准备文件
    * wiki中文数据
    [下载地址](https://dumps.wikimedia.org/zhwiki/)
    linux可用  
    ```bash
    wget https://dumps.wikimedia.org/zhwiki/20201120/zhwiki-20201120-pages-articles-multistream.xml.bz2
    ```
    下载文件需要放入ChineseWiki-master根目录,文件大小约2G
    * NER和reader的albert模型
    模型我已训练好，文件总大小约50M，下载地址
    drive:[reader](https://drive.google.com/file/d/1rQnT4j95oHkEbS5oQi6ecLkuhjzM0lRO/view?usp=sharing),[NER](https://drive.google.com/file/d/14HWqT9LDuF9kvbKFI95TziiHSI9O2BL-/view?usp=sharing)
    baiduyun：  
    * W2V
    
    
   
    
 
