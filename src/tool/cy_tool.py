# -*- coding: utf-8 -*-
from ntpath import join
import os
import jieba
import jieba.analyse
#from gensim.models import Word2Vec

Anti, Simi = True, False# Simi 同义词,Anti 反义词

class CyTool:
    def __init__(self):
        self.model = None
        self.jieba = jieba
        self.__build_dict()

    '''构词典'''
    def __build_dict(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        anti_file = os.path.join(cur_dir, 'antisem.txt')
        anti_dict = {}
        sim_dict = {}
        for line in open(anti_file, encoding='utf-8'):
            line = line.strip().split(':')
            wd = line[0]
            antis = line[1].strip().split(';')
            if wd not in anti_dict:
                anti_dict[wd] = antis
            else:
                anti_dict[wd] += antis
            for anti in antis:
                if anti not in sim_dict:
                    sim_dict[anti] = [i for i in antis if i != anti]
                else:
                    sim_dict[anti] += [i for i in antis if i != anti]
        self.__anti_dict, self.__sim_dict = anti_dict, sim_dict

    '''根据目标词获取词典近反义词'''
    def __dict_word(self, word, pattern):
        if pattern:
            return self.__anti_dict.get(word, '')
        return self.__sim_dict.get(word, '')

    '''若词典找不到词语，则进入word2vec同义词转化'''
    '''模型载入判断
    def __load_model_check(self):
        if self.model is None:
            return Word2Vec.load('src/tool/wiki_zh.model')
        else:
            return self.model
    '''
    '''
    def __model_word(self, word, pattern):
        res = []
        self.model = self.__load_model_check()
        similar_list = []
        try:
            similar_list = self.model.wv.most_similar([word])
        except KeyError:
            return []
        for n, _ in similar_list:
            # 添加近似词
            if not pattern:
                res += [n]
            # 由近义词进入词典搜索相近结果
            dict_res = self.__dict_word(n, pattern)
            if len(dict_res) != 0:
                res += dict_res
        return res
    '''
    # 获取近义词与反义词,pattern == True 反义词,pattern == False 近义词
    def __get_word(self, word, pattern):
        # 词典搜索
        dict_res = self.__dict_word(word, pattern)
        return [dict_res, [], []]

    # 分词
    def __cut_word(self, sentence):
        res = ",".join(self.jieba.cut(sentence))
        return res.split(",")

    def judge(self,word,word1):
        # 获取反义词
        anti_list = self.__get_word(word, Anti)
        #print(anti_list)
        #sim_list = self.__get_word(word, Simi)
        is_anti = self.point(anti_list,word1 )
        return is_anti
    def point(self, tup, word):
        for res in tup:
            if len(res) != 0:
                for w in res:
                    if w in word:
                        return True
        return False
'''
if __name__ == '__main__':
    k = CyTool()
    entites1=[('胸闷', '症状a', 0, 2), ('睡眠安', '正常体征', 6, 9)]
    entites2=[('胸闷', '症状a', 2, 4), ('睡眠障碍', '疾病', 7, 11), ('急性支气管炎', '疾病', 14, 20)]
    for entity1 in entites1:
        for entity2 in entites2:
            is_anti=k.judge(entity1[0],entity2[0])
            print('实体1：',entity1[0],'与实体2：',entity2[0],'，之间为：',is_anti)
    print(is_anti)
'''