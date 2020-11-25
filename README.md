# wikiCH_DRrQA
中文wiki百科问答系统，使用了的NER(ccks2016数据)和阅读理解模型(ccsk2018)，还有W2V词向量搜索。
# 模块介绍
## NER
功能：从问题中实体识别
例子：qurry:周董是谁？  》》 entiy:周董
模型：ALBERT
数据集：CCKS2016KBQA
## Word2vec
功能：如果实体不在知识库，则用W2V搜索近似实体
例子：entity:周董 >> ['周杰伦','JAY','林俊杰']
## Entity linking
功能:根据NER或W2V得到的mention entity搜索知识库
## Reader
功能：阅读理解文段，精确定位答案。
例子：参考SQuQA
# Web
功能：web服务，前端交互和结果呈现
 
