# -*- coding: utf-8 -*-
import pandas as pd
import os
df = pd.read_csv(r"wiki.csv")
df.columns = ['context','level']
df = df.drop(df[df['context']=='== 参考文献 ==\n'].index)
df = df.drop(df[df['context']=='\n'].index)
words = df[df['level']==1]['context']

try:
    os.mkdir('data')
except:
    pass

import json
from tqdm import tqdm
import re

sec = []
# total_sec = []
num_count = filename = 0
for num,data in enumerate(tqdm(zip(df['level'],df['context']))):

    level,context = data[0],data[1]
    if len(sec) == 0:
        if level == 1:
            sec = [context]
    elif len(sec)!=0:
        sec.append(context)
    if len(sec)>=4:
        # print('yes')
        sec_dict = {'id':re.sub('\n|\】|\【|\=','',''.join(sec[0]))+'|||'+str(num_count),'text':re.sub('\n|\】|\【|\=','',''.join(sec[1:]))}
        if len(sec_dict['text'])>450 or level==1:
            if level==1:
                sec = sec[:-1]
            sec_dict = {'id': re.sub('\n|\】|\【|\=', '', ''.join(sec[0])) + '|||' + str(num_count),
                        # 'text': re.sub('\n|\】|\【|\=', '', ''.join(sec[1:]))}#text是text
                        'text': re.sub('\n|\】|\【|\=', '', ''.join(sec[0]))}#text是id

            num_count += 1
#             total_sec.append(sec_dict)
#             if filename>20:
            with open(f'data//{filename}.txt','a', encoding='utf-8') as f:
                f.write(json.dumps(sec_dict, ensure_ascii=False) + '\n')
    #                 if len(total_sec)==10:
    #                     break
            sec = []
            if num_count in range(0,1000000000,20000):
                filename +=1
        else:
            pass



















# import json
# from tqdm import tqdm
# import re
#
# sec = []
# total_sec = []
# with open('zhwiki_2.txt',encoding='utf-8') as h:
#     for i in tqdm(h):
#         if len(sec) == 0:
#             if '】' in i:
#                 sec = [i]
#         if len(sec)!=0:
#             sec.append(i)
#         if len(sec)>=4:
#             sec_dict = {'id':re.sub('\n|\】|\【','',sec[0]),'text':re.sub('\n|\】|\【','',''.join(sec[1:]))}
#             if len(sec_dict['text'])>450:
#                 total_sec.append(sec_dict)
#                 filename = sec_dict['id']
#                 if len(filename) < 10:
#                     with open(f'data//{filename}.txt', 'w', encoding='utf-8') as f:
#                         f.write(json.dumps(sec_dict, ensure_ascii=False) + '\n')
#                     # if len(total_sec)==10:
#                     #     break
#                     sec = []
#                 else:
#                     sec = []
#             else:
#                 pass
