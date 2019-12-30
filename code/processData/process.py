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


# import enchant

def get_dictionary():
    dictionary = dict()
    with open("dictionary_sort_filter_2") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            dictionary[int(cache[0])] = cache[1]
    return dictionary


def get_text(dictionary):
    text_dictionary = dict()
    text_array = list()
    keys = dictionary.keys()
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            text_id = int(cache[0])
            text_list = list()
            for i in range(1, cache.__len__()):
                word_id_frequency = cache[i].split(":")
                word_id = int(word_id_frequency[0])
                word_frequency = int(word_id_frequency[1])
                if not word_id in keys:
                    continue
                word = dictionary[word_id]
                for j in range(word_frequency):
                    text_list.append(word)
            text = " ".join(text_list)
            if text == "":
                continue
            text_dictionary[text_id] = text_list
            text_array.append(text_list)
    return text_dictionary, text_array


def get_lda(text_dictionary):
    train = []

    for key, line in text_dictionary.items():
        line = line.strip().split(' ')
        train.append(line)

    print(len(train))
    print(' '.join(train[2]))

    dictionary = corpora.Dictionary(train)
    corpus = [dictionary.doc2bow(text) for text in train]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)

    topic_list = lda.print_topics(20)
    print(type(lda.print_topics(20)))
    print(len(lda.print_topics(20)))

    for topic in topic_list:
        print(topic)
    print("第一主题")
    print(lda.print_topic(1))

    print('给定一个新文档，输出其主题分布')

    # test_doc = list(new_doc) #新文档进行分词
    test_doc = train[2]  # 查看训练集中第三个样本的主题分布
    doc_bow = dictionary.doc2bow(test_doc)  # 文档转换成bow
    doc_lda = lda[doc_bow]  # 得到新文档的主题分布
    # 输出新文档的主题分布
    print(doc_lda)
    for topic in doc_lda:
        print("%s\t%f\n" % (lda.print_topic(topic[0]), topic[1]))


def get_top_frequency_word(dictionary):
    keys = dictionary.keys()
    word_frequency_dic = {key: 0 for key in keys}
    with open("jobs-unigrams") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            for i in range(1, cache.__len__()):
                word_id_frequency = cache[i].split(":")
                word_id = int(word_id_frequency[0])
                word_frequency = int(word_id_frequency[1])
                word_frequency_dic[word_id] = word_frequency_dic[word_id] + word_frequency
    word_sort = sorted(word_frequency_dic.items(), key=lambda asd: asd[1], reverse=True)
    print(word_frequency_dic)
    with open("dictionary_sort", 'w') as f:
        for item in word_sort:
            line = str(item[0]) + " " + dictionary[item[0]] + " " + str(item[1]) + '\n'
            f.write(line)
    print(word_sort)


def get_word_english_remove_stopword():
    # tag_list = ['JJ', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    tag_list = ['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS']
    stop_words = set(stopwords.words('english'))
    with open("dictionary_sort_filter_2", 'w') as w:
        with open("dictionary_sort") as f:
            for _line in tqdm(f):
                line = _line.strip().split(' ')
                # print(line)
                if line[1] in stop_words:
                    continue
                if not wordnet.synsets(line[1]):
                    # print("not an english word:",line[1])
                    pass
                else:
                    # print("english word:", line[1])
                    word_tag = pos_tag([line[1]])[0][1]
                    if word_tag in tag_list:
                        # print("english word:", line[1], word_tag)
                        w.write(_line)


def get_LDA_model(text_array):
    """
    (30, 27445)
    """
    dictionary = Dictionary(text_array)
    # print(common_dictionary)
    common_corpus = [dictionary.doc2bow(text) for text in text_array]
    # Train the model on the corpus.
    lda = LdaModel(common_corpus, id2word=dictionary, num_topics=30, passes=5, iterations=500)
    temp_file = datapath("LDA_twitter")
    lda.save(temp_file)
    topics = lda.get_topics()
    print(topics.shape)


# topic 996821183
#

def get_top_k_topic(text_dictionary, text_array):
    dictionary = Dictionary(text_array)
    temp_file = datapath("LDA_twitter")
    lda = LdaModel.load(temp_file)

    # topic_list = lda.get_topics()
    # i = 0
    # with open("twitter.content.topic", "w") as w:
    #     for topic in topic_list:
    #         # print(str(i), ":", topic)
    #         topic_list = topic.tolist()
    #         topic_list = [str(i) for i in topic_list]
    #         line = "996821183" + str(i) + "\t" + "\t".join(topic_list) + "\t" + "topic" + "\n"
    #         w.write(line)
    #         i = i + 1

    print('给定一个新文档，输出其主题分布')
    # # test_doc = list(new_doc) #新文档进行分词
    with open("twitter.topic", "w") as w:
        for key, value in text_dictionary.items():
            # 查看训练集中样本的主题分布
            doc_bow = dictionary.doc2bow(value)  # 文档转换成bow
            doc_lda = lda[doc_bow]  # 得到新文档的主题分布
            # 输出新文档的主题分布
            line = str(key) + " "
            topic_list = list()
            for topic in doc_lda:
                # print("%s\t%f" % (lda.print_topic(topic[0]), topic[1]))
                topic_list.append(str(topic[0]) + ':' + str(topic[1]))
            line = line + " ".join(topic_list) + "\n"
            w.write(line)


def transform_topic_id():
    with open("twitter.topic.top4", "w") as w:
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
                for i in range(min(4, topic_sort.__len__())):
                    new_line = new_line + "996821183" + topic_sort[i][0] + " "
                new_line = new_line + "\n"
                # print(new_line)
                w.write(new_line)


def get_unigram_word():
    with open("userId_tweet_unigram", 'w') as w:
        with open("userId_tweet") as f:
            for _line in tqdm(f):
                line = _line.strip().split(' ')
                id = line[0]
                word_set = set()
                for i in range(1, line.__len__()):
                    word_set.add(line[i])
                new_line = id + " " + " ".join(list(word_set)) + '\n'
                w.write(new_line)


def get_userId_entity():
    with open("userId_entity", 'w', encoding='utf-8') as w:
        with open("userId_tweet_unigram") as f:
            for _line in tqdm(f):
                line = _line.strip().split(' ')
                id = line[0]
                line.remove(id)
                text = " ".join(line)
                obj = Annotate(text, theta=0.2)
                entity = list()
                for i in obj.keys():
                    entity.append(i[1])
                new_line = id + "*" + "*".join(entity) + '\n'
                # print(new_line)
                w.write(new_line)


def get_twitter_content_cluster():
    dictionary = dict()
    with open("dictionary") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            dictionary[int(cache[0])] = cache[1]

    cluster = dict()
    with open("w2v-200") as f:
        for line in tqdm(f):
            cache = line.strip().split(' ')
            cluster[cache[0]] = cache[2]

    with open("twitter.content.cluster", "w") as w:
        with open("jobs-unigrams") as f:
            for line in tqdm(f):
                cache = line.strip().split(' ')
                userId = cache[0]
                cache.remove(userId)
                cluster_list = list()
                cluster_list.append(userId)
                for item in cache:
                    item = item.split(":")
                    wordId = int(item[0])
                    wordFre = int(item[1])
                    word = dictionary[wordId]
                    if word in cluster:
                        cluster_id = cluster[word]
                        for i in range(wordFre):
                            cluster_list.append(cluster_id)
                new_line = " ".join(cluster_list)
                w.write(new_line + "\n")


def get_twitter_text_no_label():
    with open("twitter.content.text.no_label", "w") as w:
        with open("twitter.content.cluster") as f:
            for line in tqdm(f):
                a = [0 for _ in range(200)]
                line = line.strip().split(" ")
                userId = line[0]
                new_line = userId + "\t"
                line.remove(userId)
                for item in line:
                    index = int(item) - 1
                    a[index] = a[index] + 1
                # print(a)
                employment = np.array(a)
                mean = employment.mean()  # 计算平均数
                deviation = employment.std()  # 计算标准差
                # 标准化数据的公式: (数据值 - 平均数) / 标准差
                standardized_employment = (employment - mean) / deviation
                standardized_employment_list = [str(i) for i in standardized_employment]
                new_line = new_line + "\t".join(standardized_employment_list) + "\n"
                w.write(new_line)


def get_twitter_text():
    occupation_dic = {"1": "Managers", "2": "Professional", "3": "Associate", "4": "Administrative", "5": "Skilled",
                      "6": "Caring", "7": "Sales", "8": "Process", "9": "Elementary"}

    userId_dic = set()
    with open("twitter.topic.top4") as f:
        for line in tqdm(f):
            userId_dic.add(line.strip().split(" ")[0])

    user_jobs_dic = dict()
    with open("jobs-users") as f:
        for line in tqdm(f):
            line = line.strip().split(" ")
            user_jobs_dic[line[0]] = line[1][0]
    # print(user_jobs_dic)

    with open("twitter.content.text", "w") as w:
        with open("twitter.content.text.no_label") as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                userId = line_list[0]
                if userId in userId_dic:
                    new_line = line.strip() + "\t" + occupation_dic[user_jobs_dic[userId]] + '\n'
                    w.write(new_line)


# 563878325
def get_entity_dic():
    entity_set = set()
    with open("userId_entity", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split("*")
            userId = line_list[0]
            line_list.remove(userId)
            for item in line_list:
                entity_set.add(item)
    entity_set.remove('')
    i = 0
    with open("entity_dic", 'w', encoding='utf-8') as w:
        for item in entity_set:
            line = "563878325" + str(i) + "\t" + item + '\n'
            w.write(line)
            i = i + 1


def get_userId_entity():
    entity_dic = dict()
    with open("entity_dic_filter", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split("\t")
            entityId = line_list[0]
            entity = line_list[1]
            entity_dic[entity] = entityId
    with open("twitter.entity", 'w', encoding='utf-8') as w:
        with open("userId_entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("*")
                userId = line_list[0]
                line_list.remove(userId)
                line_list = ["ENTITY/" + item.replace(' ', '_') for item in line_list]
                new_line = userId + " "
                id_list = list()
                for item in line_list:
                    if item in entity_dic:
                        id = entity_dic[item]
                        id_list.append(id)
                new_line = new_line + " ".join(id_list) + "\n"
                w.write(new_line)


def get_entity_simility():
    # wiki2vec = Wikipedia2Vec.load("")
    # wiki2vec.get_word_vector(u'the')
    # wiki2vec.get_entity_vector(u'Scarlett Johansson')
    # wiki2vec.most_similar(wiki2vec.get_word(u'yoda'), 5)
    # wiki2vec.most_similar(wiki2vec.get_entity(u'Scarlett Johansson'), 5)
    model = Word2Vec.load("path/to/word2vec/en.model")
    model.similarity('woman', 'man')


def get_userId_to_newId():
    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("twitter.content.text", encoding='utf-8') as f:
            i = 0
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                userId = line_list[0]
                new_line = str(i) + " " + userId + "\n"
                w.write(new_line)
                i = i + 1


def get_topicId_to_newId():
    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("twitter.content.topic", encoding='utf-8') as f:
            i = 5185
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                userId = line_list[0]
                new_line = str(i) + " " + userId + "\n"
                w.write(new_line)
                i = i + 1


def get_entityId_to_newId():
    with open("id.dictionary", 'a+', encoding='utf-8') as w:
        with open("entity_dic_filter", encoding='utf-8') as f:
            i = 5215
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                userId = line_list[0]
                new_line = str(i) + " " + userId + "\n"
                w.write(new_line)
                i = i + 1


def get_user_topic_edge():
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.topic.top4", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)


def get_user_entity_edge():
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)


def get_edge():
    # 20161
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.topic.top4", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)

    # 448327
    with open("twitter.cites", 'a+', encoding='utf-8') as w:
        with open("twitter.entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split(" ")
                userId = line_list[0]
                line_list.remove(userId)
                if line_list.__len__() == 0:
                    continue
                for item in line_list:
                    new_line = userId + "\t" + item + "\n"
                    w.write(new_line)

    # 95850
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


def get_content_text_newId():
    id_dictionary = dict()

    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            newId = line_list[0]
            oldId = line_list[1]
            id_dictionary[oldId] = newId

    with open("twitter.content.text.new", "w", encoding='utf-8') as w:
        with open("twitter.content.text", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                oldId = line_list[0]
                line_list.remove(oldId)
                new_line = id_dictionary[oldId] + "\t" + "\t".join(line_list) + "\n"
                w.write(new_line)


def get_content_topic_newId():
    id_dictionary = dict()

    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            newId = line_list[0]
            oldId = line_list[1]
            id_dictionary[oldId] = newId

    with open("twitter.content.topic.new", "w", encoding='utf-8') as w:
        with open("twitter.content.topic", encoding='utf-8') as f:
            for line in tqdm(f):
                line_list = line.strip().split("\t")
                oldId = line_list[0]
                line_list.remove(oldId)
                new_line = id_dictionary[oldId] + "\t" + "\t".join(line_list) + "\n"
                w.write(new_line)


def get_entity_word2vec():
    with open("entity.word2vec", "w", encoding='utf-8') as w:
        with open("E:\word2vec\enwiki_20180420_win10_500d.txt", encoding='utf-8') as f:
            for line in tqdm(f):
                if "ENTITY/" in line:
                    w.write(line)


def add_self_edge():
    # 0-23028
    with open("twitter.cites.new", 'a+', encoding='utf-8') as w:
        for i in range(23029):
            line = str(i) + "\t" + str(i) + "\n"
            w.write(line)


def get_twitter_entity():
    entity_dic = dict()
    with open("entity_dic_filter", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split("\t")
            entityId = line_list[0]
            entity = line_list[1]
            entity_dic[entity] = entityId

    with open("twitter.content.entity", 'w', encoding='utf-8') as w:
        with open("entity_word2vec", encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split(" ")
                entity = line[0]
                line.remove(entity)
                new_line = entity_dic[entity] + "\t" + "\t".join(line) + "\t" + "entity" + "\n"
                w.write(new_line)


def get_twitter_entity_newId():
    id_dictionary = dict()

    with open("id.dictionary", encoding='utf-8') as f:
        for line in tqdm(f):
            line_list = line.strip().split(" ")
            newId = line_list[0]
            oldId = line_list[1]
            id_dictionary[oldId] = newId

    with open("twitter.content.entity.new", "w", encoding='utf-8') as w:
        with open("twitter.content.entity", encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip().split("\t")
                entity_id = line[0]
                line.remove(entity_id)
                new_line = id_dictionary[entity_id] + "\t" + "\t".join(line) + "\n"
                w.write(new_line)


def get_train_vail_test():
    my_list = list(range(5185))
    print(my_list)
    random.shuffle(my_list)
    print(my_list)

    train_list = my_list[:2000]
    vali_list = my_list[2000:3000]
    test_list = my_list[3000:]

    with open("train.map", "w", encoding="utf-8") as w:
        for i in train_list:
            w.write(str(i) + "\n")

    with open("vali.map", "w", encoding="utf-8") as w:
        for i in vali_list:
            w.write(str(i) + "\n")

    with open("test.map", "w", encoding="utf-8") as w:
        for i in test_list:
            w.write(str(i) + "\n")

if __name__ == "__main__":
    # word_dic = get_dictionary()
    # text_dic, text_array = get_text(word_dic)
    # for key, value in text_dic.items():
    #     if value == "":
    #         print(key, ":", value)
    # print(text_array)
    # print(text_array.__len__())
    # get_topic(text_dic,text_array)
    # get_lda(text_dic)
    # get_top_frequency_word(word_dic)
    # get_LDA_model(text_array)
    # get_word_english_remove_stopword()
    # get_unigram_word()
    # get_userId_entity()
    # get_top_k_topic(text_dic, text_array)
    # transform_topic_id()
    # get_twitter_content_cluster()
    # get_twitter_text_no_label()
    # get_twitter_text()
    # get_entity_dic()
    # get_userId_entity()
    # get_userId_to_newId()
    # get_topicId_to_newId()
    # get_entityId_to_newId()
    # get_user_topic_edge()
    # get_user_entity_edge()
    # get_content_text_newId()
    # get_content_topic_newId()
    # get_entity_word2vec()
    # get_edge()
    # get_edge_newId()
    # add_self_edge()
    # get_twitter_entity()
    # get_twitter_entity_newId()
    get_train_vail_test()
    pass
