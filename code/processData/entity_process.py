from tqdm import tqdm
import codecs
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import pos_tag
from tagme_entity import Annotate
from gensim.test.utils import datapath
import numpy as np
from collections import OrderedDict


def get_entity():
    entity_set = set()
    with open("word_entity", encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip().split("\t")
            entity_set.add("ENTITY/" + line[2].replace(" ", "_"))

    print(entity_set.__len__())
    # print(entity_set)

    with open("entity_word2vec", "w", encoding='utf-8') as w:
        with open("D:\迅雷下载\enwiki_20180420_win10_500d\enwiki_20180420_win10_500d.txt", encoding='utf-8') as f:
            for line in tqdm(f):
                if "ENTITY/" in line:
                    entity = line.strip().split(" ")[0]
                    if entity in entity_set:
                        w.write(line)


def get_entity_simility():
    entity_dic = OrderedDict()
    with open("entity_word2vec", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            entity_id = line[0]
            line.remove(entity_id)
            entity_dic[entity_id] = line

    value_list = list()
    for key, value in entity_dic.items():
        value = [float(v) for v in value]
        value_list.append(value)

    with open("entity_simility", "w", encoding="utf-8") as w:
        for vec1 in value_list:
            result_list = list()
            for vec2 in value_list:
                result = cosine(vec1, vec2)
                result_list.append(result)
            result_list = [str(r) for r in result_list]
            result_list = " ".join(result_list) + "\n"
            w.write(result_list)


def cosine(_vec1, _vec2):
    _vec1 = np.array(_vec1)
    _vec2 = np.array(_vec2)
    return np.sum(_vec1 * _vec2) / (np.linalg.norm(_vec1) * np.linalg.norm(_vec2))


def cal_sim():

    with open("entity_simility", encoding="utf-8") as f:
        for line in tqdm(f):
            count_dic = {0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0}
            line = line.strip().split(" ")
            line = [float(v) for v in line]
            for v in line:
                if v > 0.1:
                    count_dic[0.1] = count_dic[0.1] + 1
                if v > 0.2:
                    count_dic[0.2] = count_dic[0.2] + 1
                if v > 0.3:
                    count_dic[0.3] = count_dic[0.3] + 1
                if v > 0.4:
                    count_dic[0.4] = count_dic[0.4] + 1
                if v > 0.5:
                    count_dic[0.5] = count_dic[0.5] + 1
                if v > 0.6:
                    count_dic[0.6] = count_dic[0.6] + 1
                if v > 0.7:
                    count_dic[0.7] = count_dic[0.7] + 1
                if v > 0.8:
                    count_dic[0.8] = count_dic[0.8] + 1
                if v > 0.9:
                    count_dic[0.9] = count_dic[0.9] + 1
            print(count_dic)



def entity_edge():

    i = 0
    with open("entity_edge", "w", encoding="utf-8") as w:
        with open("entity_simility", encoding="utf-8") as f:
            for line in tqdm(f):
                edge_list = list()
                edge_list.append(i)
                line = line.strip().split(" ")
                line = [float(v) for v in line]
                for j, value in enumerate(line):
                    if i == j:
                        continue
                    if value > 0.5:
                        edge_list.append(j)
                edge_list = ["563878325" + str(e) for e in edge_list]
                new_line = " ".join(edge_list) + "\n"
                i = i+1
                # print(i)
                w.write(new_line)

if __name__ == "__main__":
    # get_entity()
    # get_user_entity()
    # get_entity_simility()
    # cal_sim()
    # entity_edge()
    # entity_filter()
    pass
