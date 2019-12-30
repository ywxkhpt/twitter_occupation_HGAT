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
from wikipedia2vec import Wikipedia2Vec
from gensim.models import Word2Vec
import random
import collections


def get_top_frequency_word():
    """
    统计单词的频率
    :param dictionary:
    :return:
    """
    dictionary = dict()
    with open("dictionary", encoding="utf-8") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            dictionary[cache[0]] = cache[1]

    keys = dictionary.keys()
    word_frequency_dic = {key: 0 for key in keys}
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            for i in range(1, cache.__len__()):
                word_id_frequency = cache[i].split(":")
                word_id = word_id_frequency[0]
                word_frequency = int(word_id_frequency[1])
                word_frequency_dic[word_id] = word_frequency_dic[word_id] + word_frequency
    word_sort = sorted(word_frequency_dic.items(), key=lambda asd: asd[1], reverse=True)
    # print(word_frequency_dic)
    stop_words = set(stopwords.words('english'))
    # 'JJ', 'JJR', 'JJS',
    tag_list = ['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'FW', 'LS', 'VB', 'VBD', 'VBG',
                'VBN', 'VBP', 'VBZ']
    with open("dictionary_sort", 'w') as w:
        for item in word_sort:
            word = dictionary[item[0]]
            # 去除停止词
            if word in stop_words:
                continue
            # 删除数字
            if word.isdigit():
                continue
            # 去除单个字母
            if len(word) == 1:
                continue
            # 去除非英文单词
            if not wordnet.synsets(word):
                continue
            # 去除低频词
            if item[1] < 10:
                continue
            # 去除非tag_list词性的词
            word_tag = pos_tag([word])[0][1]
            if not word_tag in tag_list:
                continue
            line = str(item[0]) + " " + dictionary[item[0]] + " " + str(item[1]) + '\n'
            w.write(line)
    # print(word_sort)


def get_jobs_unigrams_filter():
    """
    过滤推文数较少的用户
    :return:
    """
    word_set = set()
    with open("dictionary_sort") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            word = line[0]
            word_set.add(word)
    print(word_set.__len__())

    user_dic = dict()
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_id = line[0]
            user_dic[user_id] = 0
            line.remove(user_id)
            for item in line:
                item = item.split(":")
                if item[0] in word_set:
                    user_dic[user_id] = user_dic[user_id] + 1

    user_sort = sorted(user_dic.items(), key=lambda asd: asd[1], reverse=True)
    with open("user_sort", 'w') as w:
        for item in user_sort:
            if item[1] < 1000:
                break
            line = item[0] + "\n"
            w.write(line)
    print(user_sort.__len__())


def get_user_label_num():
    """
    统计每一个标签的用户的数量
    :return:
    """
    user_set = set()
    with open("user_sort") as f:
        for line in tqdm(f):
            line = line.strip()
            user_set.add(line)

    label_num = {key: 0 for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']}

    with open("jobs-users-label", "w") as w:
        with open("jobs-users") as f:
            for line in tqdm(f):
                line = line.strip().split(" ")
                user_id = line[0]
                label = line[1]
                key = label[0]
                if not user_id in user_set:
                    continue
                label_num[key] = label_num[key] + 1
                w.write(user_id + " " + key + "\n")

    print(label_num)


def get_user_tweets():
    """
    获取过滤后用户的推文
    :return:
    """

    user_dic = dict()
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            line = line.strip().split(" ", 1)
            user_id = line[0]
            # print(user_id)
            # print(line[1])
            user_dic[user_id] = line[1]

    word_dictionary = dict()
    with open("dictionary_sort") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            word_id = line[0]
            word = line[1]
            word_dictionary[word_id] = word

    user_list = list()
    with open("jobs-users-label") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_id = line[0]
            user_list.append(user_id)

    with open("jobs-unigrams-filter", "w") as w:
        for user_id in user_list:
            word_list = user_dic[user_id]
            word_list = word_list.strip().split(" ")
            line_list = list()
            for item in word_list:
                item = item.split(":")
                word_id = item[0]
                freq = item[1]
                if word_id in word_dictionary:
                    word = word_dictionary[word_id]
                    for i in range(int(freq)):
                        line_list.append(word)
            line = user_id + " " + " ".join(line_list) + "\n"
            w.write(line)


def get_lda_model():
    """
    (50,28767)
    获得话题
    :return:
    """
    text_array = list()

    with open("jobs-unigrams-filter") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            line.remove(line[0])
            text_array.append(line)

    dictionary = Dictionary(text_array)
    # print(common_dictionary)
    common_corpus = [dictionary.doc2bow(text) for text in text_array]
    # Train the model on the corpus.
    lda = LdaModel(common_corpus, id2word=dictionary, num_topics=50, passes=10, iterations=1000)
    temp_file = datapath("LDA_twitter")
    lda.save(temp_file)
    topics = lda.get_topics()
    print(topics.shape)

    topic_list = lda.print_topics(50)
    for topic in topic_list:
        print(topic)


def get_tweets_topic():
    """
    (50,28767)
    :return:
    """
    text_array = list()
    text_dictionary = collections.OrderedDict()

    with open("jobs-unigrams-filter") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_id = line[0]
            line.remove(user_id)
            text_array.append(line)
            text_dictionary[user_id] = line

    dictionary = Dictionary(text_array)
    temp_file = datapath("LDA_twitter")
    lda = LdaModel.load(temp_file)

    topic_list = lda.get_topics()
    i = 0
    with open("twitter.content.topic", "w") as w:
        for topic in topic_list:
            # print(str(i), ":", topic)
            topic_list = topic.tolist()
            topic_list = [str(i) for i in topic_list]
            line = "996821183" + str(i) + "\t" + "\t".join(topic_list) + "\t" + "topic" + "\n"
            w.write(line)
            i = i + 1

    print('给定一个新文档，输出其主题分布')
    # # test_doc = list(new_doc) #新文档进行分词
    with open("twitter.topic", "w") as w:
        for key, value in text_dictionary.items():
            #
            print(key)
            # 查看训练集中样本的主题分布
            doc_bow = dictionary.doc2bow(value)  # 文档转换成bow
            doc_lda = lda[doc_bow]  # 得到新文档的主题分布
            # 输出新文档的主题分布
            line = str(key) + " "
            topic_list = list()
            for topic in doc_lda:
                # print("%s\t%f" % (lda.print_topic(topic[0]), topic[1]))
                topic_list.append("996821183" + str(topic[0]) + ':' + str(topic[1]))
            line = line + " ".join(topic_list) + "\n"
            w.write(line)


def get_top_k_topic():
    with open("twitter.topic.top2", "w") as w:
        with open("twitter.topic") as f:
            for line in tqdm(f):
                line = line.strip().split(" ")
                user_id = line[0]
                line.remove(user_id)
                topic_dic = dict()
                for item in line:
                    split = item.split(":")
                    topic_dic[split[0]] = float(split[1])
                topic_sort = sorted(topic_dic.items(), key=lambda asd: asd[1], reverse=True)
                print(topic_sort)
                new_line = user_id + " "
                for i in range(min(2, topic_sort.__len__())):
                    new_line = new_line + topic_sort[i][0] + " "
                new_line = new_line + "\n"
                # print(new_line)
                w.write(new_line)


def get_topic_id():
    li = [str(i) for i in range(50)]
    with open("topic_id", "w") as w:
        for i in li:
            w.write("996821183" + i + "\n")


def get_entity():
    """
    2934
    :return:
    """
    word_list = list()
    word_dic = dict()

    with open("dictionary_sort") as f:
        for _line in tqdm(f):
            line = _line.strip().split(' ')
            word_id = line[0]
            word = line[1]
            word_list.append(word)
            word_dic[word] = word_id

    text = " ".join(word_list)

    with open("word_entity", 'w', encoding='utf-8') as w:
        obj = Annotate(text, theta=0.2)
        # print(obj)
        for i in obj.keys():
            word = i[0]
            wordId = word_dic[word]
            entity = i[1]
            new_line = wordId + "\t" + word + "\t" + entity + "\n"
            w.write(new_line)


def get_user_entity():
    entity_dictionary = dict()

    with open("word_entity", encoding='utf-8') as f:
        for _line in tqdm(f):
            line = _line.strip().split('\t')
            word_id = line[0]
            word = line[1]
            entity = line[2]
            entity_dictionary[word_id] = entity

    user_list = list()
    with open("jobs-users-label", encoding='utf-8') as f:
        for _line in tqdm(f):
            line = _line.strip().split(' ')
            user_id = line[0]
            user_list.append(user_id)

    user_dic = dict()
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_id = line[0]
            line.remove(user_id)
            word_set = set()
            for item in line:
                item = item.split(":")
                word_set.add(item[0])
            user_dic[user_id] = word_set

    with open("twitter.entity", "w", encoding="utf-8") as w:
        for user in user_list:
            word_set = user_dic[user]
            entity_list = list()
            for word in word_set:
                if word in entity_dictionary:
                    entity_list.append(entity_dictionary[word])
            new_line = user + "\t" + "\t".join(entity_list) + "\n"
            w.write(new_line)


# 563878325
def get_entity_dic():
    entity_list = list()
    with open("entity_word2vec", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            entity_list.append(line_list[0])

    i = 0
    with open("entity_dic", 'w', encoding='utf-8') as w:
        for item in entity_list:
            line = "563878325" + str(i) + "\t" + item + '\n'
            w.write(line)
            i = i + 1


def entity_filter():
    entity_set = list()
    with open("entity_word2vec", encoding='utf-8') as f:
        for line in tqdm(f):
            entity_set.append(line.strip().split(" ")[0])

    print(entity_set.__len__())
    # print(entity_set)

    with open("word_entity_2", "w", encoding='utf-8') as w:
        with open("word_entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split("\t")
                entity = "ENTITY/" + line[2].replace(" ", "_")
                if not entity in entity_set:
                    continue
                new_line = line[0] + "\t" + line[1] + "\t" + entity + "\n"
                w.write(new_line)


def get_edge():
    # 7796
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.topic.top2", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)

    entity_dic = dict()
    with open("entity_dic", encoding="utf-8") as f:
        for line in tqdm(f):
            line = line.strip().split("\t")
            entity_dic[line[1]] = line[0]

    # 374377
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + entity_dic[item] + "\n"
                    w.write(new_line)

    # 377457
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("entity_edge", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)


def get_id_dictionary():
    i = 0

    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("jobs-users-label", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                new_line = str(i) + " " + userId + "\n"
                w.write(new_line)
                i = i + 1

    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("topic_id", encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                topicId = line
                new_line = str(i) + " " + topicId + "\n"
                w.write(new_line)
                i = i + 1

    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("entity_dic", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                entityId = line_list[0]
                new_line = str(i) + " " + entityId + "\n"
                w.write(new_line)
                i = i + 1


def get_edge_newId():
    id_dic = dict()
    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            id_dic[line_list[1]] = line_list[0]

    print(id_dic.__len__())

    with open("twitter.cites.new", 'w', encoding='utf-8') as w:
        with open("twitter.cites", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                new_line = id_dic[line_list[0]] + "\t" + id_dic[line_list[1]] + "\n"
                w.write(new_line)

    # 添加自环
    with open("twitter.cites.new", 'a+', encoding='utf-8') as w:
        for i in range(6664):
            line = str(i) + "\t" + str(i) + "\n"
            w.write(line)


def get_twitter_content_topic():
    id_dic = dict()
    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            id_dic[line_list[1]] = line_list[0]

    with open("twitter.content.topic.new", 'w', encoding='utf-8') as w:
        with open("twitter.content.topic", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t", 1)
                id = line_list[0]
                new_line = id_dic[id] + "\t" + line_list[1] + "\n"
                w.write(new_line)


def get_twitter_content_entity():
    id_dic = dict()
    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            id_dic[line_list[1]] = line_list[0]

    entity_dic = dict()
    with open("entity_dic", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split("\t")
            entity_dic[line_list[1]] = line_list[0]

    with open("twitter.content.entity.new", 'w', encoding='utf-8') as w:
        with open("entity_word2vec", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                entity = line_list[0]
                id = id_dic[entity_dic[entity]]
                line_list.remove(entity)
                line_list.append("entity")
                line_list.insert(0, id)
                new_line = "\t".join(line_list) + "\n"
                w.write(new_line)


def get_twitter_content_cluster():
    occupation_dic = {"1": "Managers", "2": "Professional", "3": "Associate", "4": "Administrative", "5": "Skilled",
                      "6": "Caring", "7": "Sales", "8": "Process", "9": "Elementary"}

    dictionary = dict()
    with open("dictionary_sort") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            dictionary[cache[1]] = cache[0]

    cluster = dict()
    with open("w2v-2000") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            cluster[cache[0]] = cache[2]

    users_label_dic = dict()
    with open("jobs-users-label") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            users_label_dic[cache[0]] = cache[1]

    with open("twitter.content.text", "w") as w:
        with open("jobs-unigrams-filter") as f:
            for line in tqdm(f):
                a = [0 for _ in range(2000)]
                cache = line.strip().split(' ')
                userId = cache[0]
                cache.remove(userId)
                cluster_list = list()
                cluster_list.append(userId)
                for word in cache:
                    if word in cluster:
                        clusterId = cluster[word]
                        index = int(clusterId) - 1
                        a[index] = a[index] + 1
                a = [str(i) for i in a]
                label = occupation_dic[users_label_dic[userId]]
                new_line = userId + "\t" + "\t".join(a) + "\t" + label + "\n"
                # print(new_line)
                w.write(new_line)


def get_twitter_content_text():
    id_dic = dict()
    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            id_dic[line_list[1]] = line_list[0]

    with open("twitter.content.text.new", "w") as w:
        with open("twitter.content.text") as f:
            for line in tqdm(f):
                line = line.strip().split("\t", 1)
                userId = line[0]
                new_line = id_dic[userId] + "\t" + line[1] + "\n"
                # print(new_line)
                w.write(new_line)


def get_train_vail_test():
    label_num = {key: 0 for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']}

    key_user_list = {key: [] for key in ['1', '2', '3', '4', '5', '6', '7', '8', '9']}

    with open("jobs-users-label") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_id = line[0]
            label = line[1]
            label_num[label] = label_num[label] + 1
            key_user_list[label].append(user_id)

    print(label_num)

    for key, value in key_user_list.items():
        random.shuffle(value)
        key_user_list[key] = value

    print(key_user_list)

    train_list = []
    vali_list = []
    test_list = []

    for key, value in key_user_list.items():
        num = label_num[key]

        train_num = int(num * 0.40)
        val_num = int(num * 0.50)

        for i in range(train_num):
            train_list.append(value[i])
        for i in range(train_num, val_num):
            vali_list.append(value[i])
        for i in range(val_num, num):
            test_list.append(value[i])

    id_dic = dict()
    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            id_dic[line_list[1]] = line_list[0]

    with open("train.map", "w", encoding="utf-8") as w:
        for i in train_list:
            w.write(str(id_dic[i]) + "\n")

    with open("vali.map", "w", encoding="utf-8") as w:
        for i in vali_list:
            w.write(str(id_dic[i]) + "\n")

    with open("test.map", "w", encoding="utf-8") as w:
        for i in test_list:
            w.write(str(id_dic[i]) + "\n")


if __name__ == "__main__":
    # get_top_frequency_word()
    # get_jobs_unigrams_filter()
    # get_user_label_num()
    # get_user_tweets()
    # get_lda_model()
    # get_tweets_topic()
    # get_top_k_topic()
    # get_entity()
    # entity_filter()
    # get_user_entity()
    # get_entity_dic()
    # entity_filter()
    # get_edge()
    # get_topic_id()
    # get_id_dictionary()
    # get_edge_newId()
    # get_twitter_content_topic()
    # get_twitter_content_entity()
    # get_twitter_content_cluster()
    # get_twitter_content_text()
    get_train_vail_test()
    pass
