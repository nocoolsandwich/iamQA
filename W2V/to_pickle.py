# -*- coding: utf-8 -*-
import gensim,pickle
model_pt_path = r'sgns.merge.word'
W2V = gensim.models.KeyedVectors.load_word2vec_format(model_pt_path,binary = False)
with open(r'W2V.pickle','wb') as f:
    pickle.dump(W2V,f)
