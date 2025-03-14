from typing import Any
import gensim.models.word2vec as w2v
from transformers import AutoTokenizer
import numpy as np
import jieba

class Tokenizer:
    def __init__(self,pretrain_path) -> None:
        self.model = w2v.Word2Vec.load(pretrain_path)
        if 'mimic' in pretrain_path:
            self.language = 'en'
        else:
            self.language = 'zh'
        self.vocab = self.model.wv.key_to_index.copy()
        self.vocab['[UNK]'] = len(self.vocab)
        self.vocab['[PAD]'] = len(self.vocab)
        self.vocab_i2w = {}
        for word in self.vocab:
            self.vocab_i2w[self.vocab[word]] = word

        self.pad_token_id = self.vocab['[PAD]']

    @classmethod
    def from_pretrained(cls,pretrain_path):
        return Tokenizer(pretrain_path)

    def __call__(self, sentence:str or list,max_length:int=None,truncation:bool=False) -> dict:
        """
        将字符串文本转化为tokens
        """
        token_ids = []
        attention_mask = []
        if type(sentence) is list:
            sentences = sentence
            for sentence in sentences:
                _token_ids = self._process(sentence)
                token_ids.append(_token_ids)
                attention_mask.append([1] * len(token_ids))

        if type(sentence) is str:
            _token_ids = self._process(sentence)
            token_ids.append(_token_ids)
            attention_mask.append([1] * len(_token_ids))

        max_sentence_len = max([len(_) for _ in token_ids])
        if max_length is None or not truncation:
            pad_len = max_sentence_len
        else:
            pad_len = min(max_length,max_sentence_len)

        for i in range(len(token_ids)):
            if len(token_ids[i]) > pad_len:
                token_ids[i] = token_ids[i][:max_length]
                attention_mask[i] = attention_mask[i][:max_length]
            else:
                token_ids[i] += [self.vocab['[PAD]']] * (pad_len-len(token_ids[i]))
                attention_mask[i] += [0] * (pad_len-len(token_ids[i]))

        if type(sentence) is list:
            return {'input_ids':token_ids,'attention_mask':attention_mask}
        else:
            return {'input_ids':token_ids[0],'attention_mask':attention_mask[0]}


    def _process(self,sentence:str):
        if 'en' in self.language:
            sentence = sentence.replace('.','').split(' ')
        else:
            sentence = jieba.cut(sentence)
        token_ids = []
        for word in sentence:
            if word == '': continue
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['[UNK]'])
        return token_ids

    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            tokens.append(self.vocab_i2w.get(int(token_id),'[UNK]'))
        return ''.join(tokens)

    def embedding_weight(self):
        """
        获取预训练权重
        """
        weights = self.model.wv.vectors.copy()
        embedding_dim = weights.shape[1] 
        unk_vector = np.random.randn(embedding_dim).reshape(1,embedding_dim)
        pad_vector = np.zeros((embedding_dim,)).reshape(1,embedding_dim)
        weights = np.concatenate((weights,unk_vector,pad_vector),axis = 0)
        return weights.astype(np.float32)
    
    def embedding_dim(self):
        return self.model.wv.vectors.shape[1]
    
import json
if __name__ == "__main__":
    """
    MIMIC数据统计
    """
    tokenizer = Tokenizer.from_pretrained('/home/lixin/work/diagnoisev3/data/mimic-3/embeds/128_0_10_cb_5n_5w.embeds')
    train_path = '/home/lixin/work/diagnoisev3/data/mimic-3/test.json'
    ans = tokenizer('this is a apple')
    weight = tokenizer.embedding_weight()
    with open(train_path,'r',encoding='utf-8')as f:
        train = json.load(f)
    lengths = []
    for item in train:
        doc = item['doc'].replace('[SEP]','').replace('[CLS]','')
        lengths.append(len(tokenizer(doc)['input_ids'][0]))
        # print(doc)
        # print(lengths[-1])
    # print(ans)
    lengths.sort()
    print()





