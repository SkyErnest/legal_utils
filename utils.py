import pandas as pd
import numpy as np
import json, re, pickle, os, time, multiprocessing, random
import jieba
import tensorflow as tf
from collections import Iterable
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from scipy import sparse
from tqdm import tqdm
import gensim
import thulac
from stanfordcorenlp import StanfordCoreNLP


# def read_data(prefix='train'):
#     dicts, accu_data ,law_data = [], [], []
#     with open('data/data_{}.json'.format(prefix), 'r', encoding='utf-8') as f:
#         for line in f:
#             data = json.loads(line)
#             dicts.append(data)
#             accu_data.append([data['fact'], data['meta']['accusation']])
#             law_data.append([data['fact'], data['meta']['relevant_articles']])
#     return dicts,accu_data,law_data

def count_words(accu_data):
    word_set = set()
    fact_lists = []
    for fact, accus in accu_data:
        cut_res = set(jieba.cut(fact))
        word_set |= cut_res
        fact_lists.append(list(cut_res))
    return word_set, fact_lists


def count_idf(accu_data):
    word_set, fact_lists = count_words(accu_data)
    freq = {}
    for fact in fact_lists:
        for word in fact:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
    idf = {}
    for word in word_set:
        idf[word] = len(fact_lists) / (freq[word] + 1)
    return idf


def gen_idf_file(idf):
    with open('data/idf', 'w', encoding='utf-8') as f:
        for word in idf:
            f.writelines('{}\t{}\n'.format(word, idf[word]))


def count_accu_data(accu_data):
    group = set()
    accu_list = []
    accu_freq = {}
    group_accu_freq = {}
    group_freq = {}
    for each in accu_data:
        accus = each[1]
        if len(accus) > 1:
            group.add(tuple(accus))
            if tuple(accus) not in group_freq:
                group_freq[tuple(accus)] = 0
            group_freq[tuple(accus)] += 1
        for each_accu in accus:
            if len(accus) > 1:
                if each_accu not in group_accu_freq:
                    group_accu_freq[each_accu] = 0
                group_accu_freq[each_accu] += 1
            if each_accu not in accu_list:
                accu_list.append(each_accu)
                accu_freq[each_accu] = 0
            accu_freq[each_accu] += 1
    # print(accu_list)
    # print(group)
    # print(len(accu_list))
    # print(sorted(accu_freq.items(),key=lambda x:x[1],reverse=True))
    # print(sorted(group_accu_freq.items(), key=lambda x:x[1], reverse=True))
    group4classification = sorted(group_freq.items(), key=lambda x: x[1], reverse=True)
    print(group4classification)
    group4classification = [x[0] for x in group4classification if x[1] > 100]
    group4classification = sorted(group4classification, key=lambda x: len(x), reverse=True)
    with open('data/accu_group.txt', 'w', encoding='utf-8') as f:
        for each in group4classification:
            f.writelines(str(each) + '\n')


def look_up_laws(law):
    def look_up(law):
        if 102 <= law <= 113:
            return 0
        if 114 <= law <= 139:
            return 1
        if 140 <= law <= 231:
            return 2
        if 232 <= law <= 262:
            return 3
        if 263 <= law <= 276:
            return 4
        if 277 <= law <= 367:
            return 5
        if 268 <= law <= 381:
            return 6
        if 382 <= law <= 396:
            return 7
        if 397 <= law <= 419:
            return 8
        if 420 <= law <= 451:
            return 9

    res = []
    for each in law:
        res.append(look_up(each))
    return res


def convert_to_chapter(law_data):
    return list(map(lambda x: [x[0], look_up_laws(x[1])], law_data))


# def cut_data(law_data):
#     context,label,n_words=[],[],[]
#     for each in law_data:
#         context.append(' '.join(jieba.cut(each[0])))
#         n_words.append(len(context[-1]))
#         label.append(each[1])
#     return context,label,n_words


# def index_to_label(index,batch_size):
#     # convert batch index of one-hot to label
#     res=[[] for i in range(batch_size)]
#     for each in index:
#         res[int(each[0])].append(int(each[1]))
#     return res

def law_to_list(path, remain_new_line=False):
    with open(path, 'r', encoding='utf-8') as f:
        law = []
        for line in f:
            if line == '\n' or re.compile(r'第.*[节|章]').search(line[:10]) is not None:
                continue
            try:
                tmp = re.compile(r'第.*条').search(line.strip()[:8]).group(0)
                if remain_new_line:
                    law.append(line)
                else:
                    law.append(line.strip())
            except (TypeError, AttributeError):
                if remain_new_line:
                    law[-1] += line
                else:
                    law[-1] += line.strip()
    return law


def cut_law(law_list, order=None, cut_sentence=False, cut_penalty=False, stop_words_filtered=True):
    res = []
    cut = get_cutter(stop_words_filtered=stop_words_filtered)
    if order is not None:
        filter = order.keys()
    for each in law_list:
        index, content = each.split('　')
        index = hanzi_to_num(index[1:-1])
        charge, content = content[1:].split('】')
        # if charge[-1]!='罪':
        #     continue
        if order is not None and index not in filter:
            continue
        if cut_penalty:
            context, n_words = process_law(content, cut)
        elif cut_sentence:
            context, n_words = [], []
            for i in content.split('。'):
                if i != '':
                    context.append(cut(i))
                    n_words.append(len(context[-1]))
        else:
            context = cut(content)
            n_words = len(context)
        res.append([index, charge, context, n_words])
    if order is not None:
        res = sorted(res, key=lambda x: order[x[0]])
    return res


# def hanzi_to_num(hanzi):
#     # for num<10000
#     hanzi=hanzi.strip().replace('零', '')
#     if(hanzi[0])=='十': hanzi = '一'+hanzi
#     d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,'':0}
#     m = {'十':1e1,'百':1e2,'千':1e3}
#     res = 0
#     tmp=0
#     for i in range(len(hanzi)):
#         if hanzi[i] in d:
#             tmp+=d[hanzi[i]]
#         else:
#             tmp*=m[hanzi[i]]
#             res+=tmp
#             tmp=0
#     return int(res+tmp)

def read_data(prefix='train'):
    main_data = []
    with open('data/{}_cs.json'.format(prefix), 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            main_data.append([data['fact_cut'], data['accu'], data['law'],
                              data['time'], data['term_cate'], data['term']])

        return main_data


def get_tensor_shape(T):
    return T.get_shape().as_list()


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# print(stopwordslist('data//stop_word.txt'))

def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '').replace('两', '二').replace('多', '五').replace('几', '五')
    if hanzi == '':
        return 0
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            if hanzi[-1] in w:
                try:
                    return int(float(hanzi[:-1]) * w[hanzi[-1]])
                except ValueError:
                    return 0

            try:
                return int(float(hanzi))
            except (ValueError, OverflowError):
                return 0

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)


def get_cutter(dict_path="data/Thuocl_seg.txt", mode='thulac', stop_words_filtered=True):
    if stop_words_filtered:
        stopwords = stopwordslist('data/stop_word.txt')  # 这里加载停用词的路径
    else:
        stopwords = []
    if mode == 'jieba':
        jieba.load_userdict(dict_path)
        return lambda x: [a for a in list(jieba.cut(x)) if a not in stopwords]
    elif mode == 'thulac':
        thu = thulac.thulac(user_dict=dict_path, seg_only=True)
        return lambda x: [a for a in thu.cut(x, text=True).split(' ') if a not in stopwords]


def seg_sentence(sentence, cut):
    # cut=get_cutter()
    # sentence_seged = thu.cut(sentence.strip(), text=True).split(' ')
    sentence_seged = cut(sentence)
    # print(sentence_seged)
    outstr = []
    for word in sentence_seged:
        if word != '\t':
            word = str(hanzi_to_num(word))
            outstr.append(word)
            # outstr += " "
    return outstr


def process_penalty(data):
    v = data['imprisonment']
    if data["death_penalty"]:
        opt = 0
    elif data["life_imprisonment"]:
        opt = 11
    elif v > 10 * 12:
        opt = 1
    elif v > 7 * 12:
        opt = 2
    elif v > 5 * 12:
        opt = 3
    elif v > 3 * 12:
        opt = 4
    elif v > 2 * 12:
        opt = 5
    elif v > 1 * 12:
        opt = 6
    elif v > 9:
        opt = 7
    elif v > 6:
        opt = 8
    elif v > 0:
        opt = 9
    else:
        opt = 10

    return opt


def process_penalty2(term_cate, term):
    if term_cate == 0:
        return term_cate
    if term_cate == 1:
        return 11
    else:
        return term


def cut_data(data, cut_sentence=True, cut_word=True, stop_words_filtered=True):
    # cut = get_cutter(stop_words_filtered=stop_words_filtered)
    context, accu_label, law_label, imprisonments, n_words = [], [], [], [], []
    for each in data:
        if cut_sentence:
            sent_words, sent_n_words = [], []
            for i in each[0].split('。'):
                if i != '':
                    if cut_word:
                        sent_words.append(seg_sentence(i, cut))
                        # sent_words.append(list(jieba.cut(i)))
                    else:
                        sent_words.append(i)
                    sent_n_words.append(len(sent_words[-1]))
            context.append(sent_words)
            n_words.append(sent_n_words)
        else:
            if cut_word:
                context.append(cut(each[0]))
            else:
                context.append(each[0])
            n_words.append(len(context[-1]))
        accu_label.append([i.replace('[', '').replace(']', '') for i in each[1]])
        law_label.append(each[2])
        imprisonments.append(process_penalty(each[3]))
    return context, accu_label, law_label, imprisonments, n_words


def load_law(word2id=None, law_doc_len=None, law_sent_len=None, law_data_path='data/law_cut.pkl',
             law_path='data/criminal_law.txt', law_list_path='data/new_law.txt'):
    with open(law_list_path, 'r', encoding='utf-8') as f:
        law_class = [int(i) for i in f.read().split('\n')[:-1]]
        law_file_order = {law_class[i]: i for i in range(len(law_class))}
        n_law = len(law_class)  # 183

    # if os.path.exists(law_data_path):
    #     laws, laws_doc_length, laws_sent_length = pickle.load(open(law_data_path, 'rb'))
    # else:
    law_list = law_to_list(law_path)
    laws = cut_law(law_list, order=law_file_order, cut_sentence=True)

    laws = list(zip(*laws))
    # pickle.dump({laws[1][i]:laws[0][i] for i in range(len(laws[0]))},open('data/accu2law_dict.pkl','wb'))
    # law_set=laws[0]
    laws_doc_length = [len(i) if len(i) < law_doc_len else law_doc_len for i in laws[-1]]
    laws_sent_length = trun_n_words(laws[-1], law_sent_len)
    laws_sent_length = align_flatten2d(laws_sent_length, law_doc_len, flatten=False)
    laws = lookup_index_for_sentences(laws[-2], word2id, law_doc_len, law_sent_len)

    # pickle.dump((laws, laws_doc_length, laws_sent_length), open(law_data_path, 'wb'))
    return laws, law_class, laws_doc_length, laws_sent_length


def gen_w2id(words, frequency=25):
    count = {}
    for each in words:
        if each not in count:
            count[each] = 1
        else:
            count[each] += 1
    words = [each for each in count if count[each] >= frequency]
    word2id = dict(zip(words, np.arange(len(words))))
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)
    return word2id


def load_embeddings(emb_path, w2id_path, w2v_model_path):
    model = gensim.models.Word2Vec.load(w2v_model_path)
    vec = model.wv.vectors
    vec = np.concatenate([vec, np.zeros([2, vec.shape[-1]], np.float32)], 0)
    np.save(emb_path, vec)
    w2id = {w: i for i, w in enumerate(model.wv.index2word)}
    w2id['UNK'] = len(w2id)
    w2id['BLANK'] = len(w2id)
    pickle.dump(w2id, open(w2id_path, 'wb'))


def load_embeddings2(emb_path='data/cail_jieba.npy', w2id_path='data/w2id_jieba.pkl', w2v_model_path=None):
    if not (os.path.exists(w2id_path) and os.path.exists(emb_path)):
        load_embeddings(emb_path, w2id_path, w2v_model_path)
    emb = np.load(emb_path)
    w2id = pickle.load(open(w2id_path, 'rb'))
    return emb, w2id


def load_data(batch_size, num_epochs=0, batch=True, doc_len=None, sent_len=None,
              data_path='data/legal_data.pkl',
              accu_path='data/new_accu.txt', index_filter=False):
    data_filter_path = 'data/index_filter.pkl'

    with open(accu_path, 'r', encoding='utf-8')as f:
        accu_class = [i for i in f.read().split('\n')[:-1]]
        accu_file_order = {accu_class[i]: i for i in range(len(accu_class))}
        n_accu = len(accu_class)  # 202

    if os.path.exists(data_path):
        batches_train, batches_val, batches_test = pickle.load(open(data_path, 'rb'))
    else:
        batches_train, batches_val, batches_test = process(doc_len, sent_len)
        # batches_train=batches_val=batches_test = process('valid')
        pickle.dump([batches_train, batches_val, batches_test], open(data_path, 'wb'))

    if index_filter:
        index_filter = pickle.load(open(data_filter_path, 'rb'))
        batches_train = np.array(batches_train)[index_filter['train']]
        batches_val = np.array(batches_val)[index_filter['valid']]
        batches_test = np.array(batches_test)[index_filter['test']]
    else:
        index_filter = {'train': None, 'valid': None, 'test': None}
        # global self.train_step_per_epoch, self.val_step_per_epoch, self.test_step_per_epoch

    train_step_per_epoch = int((len(batches_train) - .1) / batch_size) + 1
    val_step_per_epoch = int((len(batches_val) - .1) / batch_size) + 1
    test_step_per_epoch = int((len(batches_test) - .1) / batch_size) + 1
    if batch:
        batches_train = batch_iter(batches_train, batch_size, num_epochs)
        batches_val = list(batch_iter(batches_val, batch_size, 1, shuffle=False))
        batches_test = list(batch_iter(batches_test, batch_size, 1, shuffle=False))

    # for batch in batches
    # data, accu, law, n_sent, n_words = list(zip(*batch))
    # data:[batch_size,n_sent,n_words]
    return batches_train, batches_val, batches_test, index_filter, accu_class, train_step_per_epoch, val_step_per_epoch, test_step_per_epoch


def load_data2(batch_size, word2id, num_epochs=0, batch=True, doc_len=15, sent_len=90, cut_sentence=False,
               data_path='data/legal_data', dataset_suffix=''):
    data_path += f'_{doc_len}-{sent_len}-cut_sent-{cut_sentence}.pkl'
    if os.path.exists(data_path):
        # batches_train, batches_val, batches_test = pickle.load(open(data_path, 'rb'))
        batches = pickle.load(open(data_path, 'rb'))
    else:
        # batches_train, batches_val, batches_test = process(doc_len, sent_len, word2id=word2id,
        #                                                    dataset_suffix=dataset_suffix, cut_sentence=cut_sentence)
        batches = process(doc_len, sent_len, word2id=word2id, dataset_suffix=dataset_suffix, cut_sentence=cut_sentence)
        # batches_train=batches_val=batches_test = process('valid')
        # pickle.dump([batches_train, batches_val, batches_test], open(data_path, 'wb'))
        pickle.dump(batches, open(data_path, 'wb'))
        # global self.train_step_per_epoch, self.val_step_per_epoch, self.test_step_per_epoch

    # train_step_per_epoch = int((len(batches_train) - .1) / batch_size) + 1
    # val_step_per_epoch = int((len(batches_val) - .1) / batch_size) + 1
    # test_step_per_epoch = int((len(batches_test) - .1) / batch_size) + 1
    step_per_epoch = [int((len(each_batches) - .1) / batch_size) + 1 for each_batches in batches]
    if batch:
        # batches_train = batch_iter(batches_train, batch_size, num_epochs)
        # batches_val = list(batch_iter(batches_val, batch_size, 1, shuffle=False))
        # batches_test = list(batch_iter(batches_test, batch_size, 1, shuffle=False))
        batches_train = batch_iter(batches[0], batch_size, num_epochs)
        batches = [batches_train] + [list(batch_iter(each_batches, batch_size, 1, shuffle=False)) for each_batches in
                                     batches[1:]]

    # for batch in batches
    # data, law, accu, n_sent, n_words = list(zip(*batch))
    # data:[batch_size,n_sent,n_words]
    # return batches_train, batches_val, batches_test, train_step_per_epoch, val_step_per_epoch, test_step_per_epoch
    return batches + step_per_epoch


def preprocess(param='train', cut_sentence=False, stop_words_filtered=False):
    data_path = 'data/thulac_cut_' + param + '-sent-{}-stop_words-{}.pkl'.format(cut_sentence, stop_words_filtered)
    if os.path.exists(data_path):
        x, accu, law, imprisonments, n_words = pickle.load(open(data_path, 'rb'))
    else:
        data_train = read_data(param)
        x, accu, law, imprisonments, n_words = cut_data(data_train, cut_sentence=cut_sentence,
                                                        stop_words_filtered=stop_words_filtered)
        pickle.dump([x, accu, law, imprisonments, n_words], open(data_path, 'wb'))

    return x, accu, law, imprisonments, n_words


def process(doc_len, sent_len, cut_sentence=False, tfidf_path='data/tfidf2.model', word2id=None, dataset_suffix=''):
    x_train, law_train, accu_train, imprisonments_train, n_words_train = read_data2('train' + dataset_suffix,
                                                                                    cut_sentence)
    if dataset_suffix != '_big':
        x_val, law_val, accu_val, imprisonments_val, n_words_val = read_data2('valid' + dataset_suffix, cut_sentence)

    # if word2id==None:
    #     if dataset_suffix != '_big':
    #         words = flatten(x_train) + flatten(x_val)
    #     else:
    #         words = flatten(x_train)
    #     word2id = gen_w2id(words)
    #     pickle.dump(word2id, open(f'data/new_w2id{dataset_suffix}.pkl','wb'))

    x_test, law_test, accu_test, imprisonments_test, n_words_test = read_data2('test' + dataset_suffix, cut_sentence)

    def ending_process(x, law, accu, imprisonments, n_words):
        if cut_sentence:
            space_context = [' '.join(flatten(each)) for each in x]
            n_sent = [len(i) if len(i) < doc_len else doc_len for i in n_words]
            n_words = trun_n_words(n_words, sent_len)
            n_words = align_flatten2d(n_words, doc_len, flatten=False)
            x = lookup_index_for_sentences(x, word2id, doc_len, sent_len)
        else:
            n_sent = n_words
            space_context = [' '.join(each) for each in x]
            x = lookup_index(x, word2id, doc_len)
        # x=[[word2id[j] for j in i.split()] for i in x]
        TFIDF = pickle.load(open(tfidf_path, 'rb'))
        tfidf = TFIDF.transform(space_context)
        batches = list(zip(x, tfidf, law, accu, imprisonments, n_sent, n_words))

        return batches

    if dataset_suffix != '_big':

        return ending_process(x_train, law_train, accu_train, imprisonments_train, n_words_train), \
               ending_process(x_val, law_val, accu_val, imprisonments_val, n_words_val), \
               ending_process(x_test, law_test, accu_test, imprisonments_test, n_words_test)
    else:
        return ending_process(x_train, law_train, accu_train, imprisonments_train, n_words_train), \
               [1] * 1000, \
               ending_process(x_test, law_test, accu_test, imprisonments_test, n_words_test)


def read_data2(prefix='train', cut_sentence=False, cut_word=True):
    context, law_label, accu_label, imprisonments, n_words = [], [], [], [], []
    with open(f'data/{prefix}_cs.json', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if cut_sentence:
                sent_words, sent_n_words = [], []
                for i in data['fact_cut'].split('。'):
                    if i != '':
                        if cut_word:
                            sent_words.append(i.strip().split(' '))
                            # sent_words.append(list(jieba.cut(i)))
                        else:
                            sent_words.append(i)
                        sent_n_words.append(len(sent_words[-1]))
                context.append(sent_words)
                n_words.append(sent_n_words)
            else:
                if cut_word:
                    context.append(data['fact_cut'].strip().split(' '))
                else:
                    context.append(data['fact_cut'])
                n_words.append(len(context[-1]))
            law_label.append(data['law'])
            accu_label.append(data['accu'])
            imprisonments.append(process_penalty2(data['term_cate'], data['term']))
        return context, law_label, accu_label, imprisonments, n_words


def get_w2id(w2id_path, dataset_suffix):
    if w2id_path is None or not os.path.exists(w2id_path):
        x_train, law_train, accu_train, imprisonments_train, n_words_train = read_data2('train_big',
                                                                                        cut_sentence=False)
        # x_val, law_val, accu_val, imprisonments_val, n_words_val = utils.read_data2('valid', cut_sentence=False)

        words = flatten(x_train)  # + utils.flatten(x_val)
        if dataset_suffix == '_big':
            x_val, law_val, accu_val, imprisonments_val, n_words_val = read_data2('valid', cut_sentence=False)
            words += flatten(x_val)
        word2id = gen_w2id(words)
        pickle.dump(word2id, open(w2id_path, 'wb'))
    else:
        word2id = pickle.load(open(w2id_path, 'rb'))

        return word2id


def get_feed_dict(batch, batch_size):
    num_fix = len(batch)
    while len(batch) < batch_size:
        batch = np.concatenate([batch, batch[:batch_size - len(batch)]])
    x, tfidf, law, accu, imprisonments, n_sent, n_words = list(zip(*batch))
    tfidf = sparse.vstack(tfidf)
    # law_index= np.argsort(-self.SVM.decision_function(tfidf),-1)[:,:20]
    return x, tfidf, law, accu, imprisonments, n_sent, n_words, num_fix


def onehot(indices, vec_len):
    labels = np.zeros([len(indices), vec_len], dtype=np.int32)
    labels[list(zip(*enumerate(indices)))] = 1
    return labels


# def cut_data_in_sentence(law_data):
#     context,label,n_sents,n_words=[],[],[],[]
#     for each in law_data:
#         sent_words,sent_n_words=[],[]
#         for i in each[0].split('。'):
#             sent_words.append(list(jieba.cut(i)))
#             sent_n_words.append(len(sent_words[-1]))
#         context.append(sent_words)
#         n_sents.append(len(context[-1]))
#         n_words.append(sent_n_words)
#         label.append(each[1])
#     return context,label,n_words,n_sents

def lookup_index(x, word2id, doc_len):
    res = []
    for each in x:
        tmp = [word2id['BLANK']] * doc_len
        for i in range(len(each)):
            if i >= doc_len:
                break
            try:
                tmp[i] = word2id[each[i]]
            except KeyError:
                tmp[i] = word2id['UNK']
        res.append(tmp)
    return np.array(res)


def lookup_index_for_sentences(x, word2id, doc_len, sent_len):
    res = []
    for each in x:
        # tmp = [[word2id['BLANK']] * sent_len for _ in range(doc_len)]
        tmp = lookup_index(each, word2id, sent_len)[:doc_len]
        # tmp=np.pad(tmp,pad_width=[[0,doc_len-len(tmp)],[0,0]],mode='constant')
        tmp = np.concatenate([tmp, word2id['BLANK'] * np.ones([doc_len - len(tmp), sent_len], dtype=np.int)], 0)
        res.append(tmp)
    return np.array(res)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
	Generates a batch iterator for a dataset.
	"""
    data = np.array(data)
    data_size = len(data)
    # Original
    # num_batches_per_epoch = (int)(round(len(data)/batch_size))
    num_batches_per_epoch = int(((len(data) - .5) / batch_size) + 1)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def attention(Q, K, V):
    # Q ...*N*F
    # K ...*M*F
    # V ...*M*L
    res_map = Q @ tf.transpose(K, [-1, -2]) / tf.sqrt(K.get_shape().as_list()[-1])
    res = tf.nn.softmax(res_map) @ V
    return res


def multihead_atten(Q, K, V, F_=None, L_=None, num_attention_heads=1, initializer_range=.02):
    # Q ...*N*F
    # K ...*M*F
    # V ...*M*F
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    Q_shape = get_tensor_shape(Q)
    K_shape = get_tensor_shape(K)
    V_shape = get_tensor_shape(V)
    initializer = tf.truncated_normal_initializer(initializer_range)
    if F_ is None:
        F_ = Q_shape[-1]
    if L_ is None:
        L_ = V_shape[-1]
    Q_ = tf.layers.dense(Q, num_attention_heads * F_, activation=None, kernel_initializer=initializer)
    K_ = tf.layers.dense(K, num_attention_heads * F_, activation=None, kernel_initializer=initializer)
    V_ = tf.layers.dense(V, num_attention_heads * L_, activation=None, kernel_initializer=initializer)

    Q_ = transpose_for_scores(Q_, Q_shape[0], num_attention_heads, Q_shape[-2], F_)
    K_ = transpose_for_scores(K_, K_shape[0], num_attention_heads, K_shape[-2], F_)
    V_ = transpose_for_scores(V_, V_shape[0], num_attention_heads, V_shape[-2], L_)

    return tf.reshape(tf.transpose(attention(Q_, K_, V_), [0, 2, 1, 3]),
                      [V_shape[0], Q_shape[1], num_attention_heads * L_])


def index_to_label(index, batch_size):
    # convert batch index of one-hot to label
    res = [[] for i in range(batch_size)]
    for each in index:
        res[int(each[0])].append(int(each[1]))
    return np.array(res)


def flatten(x):
    return [y for l in x for y in flatten(l)] if type(x) is list else [x]


def flatten2(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten2(x)
        else:
            yield x


def count(x):
    res = {}
    for each in x:
        if each not in res:
            res[each] = 1
        else:
            res[each] += 1
    return res


def align_flatten2d(items, align_len, flatten=True):
    res = []
    for each in items:
        each = each[:align_len]
        res.append(np.pad(each, [0, align_len - len(each)], 'constant'))
    res = np.array(res)
    if flatten:
        res = res.flatten()
    return res


def trun_n_words(n_words, sent_len):
    for i in range(len(n_words)):
        for j in range(len(n_words[i])):
            if n_words[i][j] > sent_len:
                n_words[i][j] = sent_len
    return n_words


def find_1_in_one_hot(matrix, f):
    for each in matrix:
        for i in range(len(each)):
            if each[i] == 1:
                f(i)


def process_law(law, cut):
    # single article
    # cut=get_cutter()
    condition_list = []
    for each in law.split('。')[:-1]:
        suffix = None
        if '：' in each:
            each, suffix = each.split('：')
            suffix = cut(suffix)
        words = cut(each)
        seg_point = [-1]
        conditions = []

        for i in range(len(words)):
            if words[i] == '；' or words[i] == ';':
                seg_point.append(i)
        seg_point.append(len(words))
        for i in range(len(seg_point) - 1):
            for j in range(seg_point[i + 1] - 1, seg_point[i], -1):
                if j + 1 < len(words) and words[j] == '的' and words[j + 1] == '，':
                    conditions.append(words[seg_point[i] + 1:j + 1])
                    break
        # context=law.split('。')[:-1]
        for i in range(1, len(conditions)):
            conditions[i] = conditions[0] + conditions[i]
        # if len(condition_list)==0 and len(conditions)==0:
        #     conditions.append([])
        if suffix is not None:
            conditions = [x + suffix for x in conditions]
        condition_list += conditions

    if condition_list == []:
        condition_list.append(cut(law[:-1]))
    n_word = [len(i) for i in condition_list]
    return condition_list, n_word


def gen_tfidf_vec(accu_file_order, max_features):
    tfidf = TfidfVectorizer(max_features=max_features)
    data_train = read_data('train')
    x, accu, law, imprisonments, n_words = cut_data(data_train, cut_sentence=False)
    x = [' '.join(each) for each in x]

    def tfidf_by_class(x, accu):
        x_list = ['' for _ in range(len(accu_file_order))]
        for i in range(len(x)):
            x_list[accu_file_order[accu[i][0]]] += ' ' + x[i]
        return x_list

        # tfidf_by_class(x,accu)

    a = tfidf.fit_transform(tfidf_by_class(x, accu))
    pickle.dump(a.transpose(), open('data/tfidf_wordvec_{}'.format(max_features), 'wb'))
    pickle.dump(tfidf.vocabulary_, open('data/tfidf_w2id_{}'.format(max_features), 'wb'))


def cail_data_filter():
    with open('data/law.txt', 'r', encoding='utf-8') as f:
        law_class = [int(i) for i in f.read().split('\n')[:-1]]

    with open('data/accu.txt', 'r', encoding='utf-8')as f:
        accu_class = [i for i in f.read().split('\n')[:-1]]

    dicts_train, data_train = read_data('train')
    x_train, accu_train, law_train, imprisonments, n_words_train = cut_data(data_train, cut_sentence=False,
                                                                            cut_word=False)
    dicts_valid, data_val = read_data('valid')
    x_val, accu_val, law_val, imprisonments, n_words_val = cut_data(data_val, cut_sentence=False, cut_word=False)
    dicts_test, data_test = read_data('test')
    x_test, accu_test, law_test, imprisonments, n_words_test = cut_data(data_test, cut_sentence=False, cut_word=False)

    law_count = {i: 0 for i in law_class}
    accu_count = {i: 0 for i in accu_class}

    def get_count(x, accu_train, law_train):
        index_train = []
        for i in range(len(law_train)):
            if len(law_train[i]) > 1 or len(accu_train[i]) > 1 or '二审' in x[i]:
                index_train.append(i)
            else:
                law_count[law_train[i][0]] += 1
                accu_count[accu_train[i][0]] += 1
        return index_train

    index_train = get_count(x_train, accu_train, law_train)
    index_val = get_count(x_val, accu_val, law_val)
    law_class = [i for i in law_class if law_count[i] >= 100]
    accu_class = [i for i in accu_class if accu_count[i] >= 100]
    index_test = get_count(x_test, accu_test, law_test)

    print(len(law_class), len(accu_class))

    with open('data/law_filtered.txt', 'w', encoding='utf-8') as f:
        for each in law_class:
            f.write(str(each))
            f.write('\n')
    with open('data/accu_filtered.txt', 'w', encoding='utf-8') as f:
        for each in accu_class:
            f.write(str(each))
            f.write('\n')

    def get_index(accu_test, law_test, index_test):
        for i in range(len(law_test)):
            if accu_test[i][0] not in accu_class or law_test[i][0] not in law_class:
                index_test.append(i)
        return [i for i in range(len(law_test)) if i not in index_test]

    index_train = get_index(accu_train, law_train, index_train)
    index_val = get_index(accu_val, law_val, index_val)
    index_test = get_index(accu_test, law_test, index_test)
    print(len(index_train), len(index_val), len(index_test))
    pickle.dump({'train': index_train, 'valid': index_val, 'test': index_test}, open('data/index_filter.pkl', 'wb'))


def cos_similarity(a, b):
    return np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b))


def gen_law_relation(law_list_path='data/law.txt'):
    with open(law_list_path, 'r', encoding='utf-8') as f:
        law_class = [int(i) for i in f.read().split('\n')[:-1]]
        law_file_order = {law_class[i]: i for i in range(len(law_class))}
        n_law = len(law_class)  # 183

    law_list = law_to_list('data/criminal_law.txt')
    laws = cut_law(law_list, order=law_file_order, cut_penalty=True)
    laws = list(zip(*laws))
    index = laws[0]
    laws = [' '.join(flatten(each)) for each in laws[-2]]
    tfidf = TfidfVectorizer().fit_transform(laws).toarray()

    sim = np.zeros([n_law, n_law])
    for i in range(n_law):
        for j in range(n_law):
            sim[i, j] = cos_similarity(tfidf[i], tfidf[j])
    return sim  # proper threshold 0.3


def get_subgraph(neigh_index_dict):
    graph = []
    items = []
    graph_ship = {}
    for i in range(len(neigh_index_dict)):
        if len(neigh_index_dict[i]) == 0:
            graph.append([i])
        else:
            if neigh_index_dict[i][0] in items:
                continue
            else:
                sub_graph = neigh_index_dict[i]
                finding = neigh_index_dict[i]
                exchange = []
                for j in finding:
                    exchange += neigh_index_dict[j]
                exchange = list(set(exchange))  # 去重
                finding = exchange
                exchange = []
                while (set(sub_graph) >= set(finding)) is False:
                    sub_graph = list(set(sub_graph).union(set(finding)))
                    for j in finding:
                        exchange += neigh_index_dict[j]
                    exchange = list(set(exchange))
                    finding = exchange
                    exchange = []
                graph.append(sub_graph)
                items += sub_graph
    for i in range(len(graph)):
        graph_1 = {j: i for j in graph[i]}
        graph_ship.update(graph_1)
    graph_ship = sorted(graph_ship.items())

    return graph, graph_ship


def score(label, pred):
    acc = metrics.accuracy_score(label, pred)
    precision = metrics.precision_score(label, pred, average='macro')
    recall = metrics.recall_score(label, pred, average='macro')
    f1 = metrics.f1_score(label, pred, average='macro')
    return acc, precision, recall, f1


def scatter(x, colors, n_classes):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n_classes))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(n_classes):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def corenlp_sent(nlp, sent, props):
    r_dict = json.loads(nlp.annotate(sent, props))
    words = [token['originalText'] for s in r_dict['sentences'] for token in s['tokens']]
    words_ = []
    tags = []
    for s in r_dict['sentences']:
        for token in s['tokens']:
            words_.append(token['originalText'])
            tags.append(token['pos'])
    tags = list(zip(words_, tags))
    dependencies = [(dep['dep'], dep['governor'], dep['dependent']) for s in r_dict['sentences'] for dep in
                    s['basicDependencies']]
    # words = nlp.word_tokenize(sent)
    # tags = nlp.pos_tag(sent)
    # dependencies = nlp.dependency_parse(sent)
    return words, tags, dependencies


def get_collocations(file_path, out_path):
    print('Time:', time.asctime(), 'starting to get collocations in', file_path)
    with StanfordCoreNLP('../stanford-corenlp-full-2018-10-05', lang='zh') as nlp:
        props = {'annotators': 'tokenize,ssplit,pos,depparse', 'pipelineLanguage': 'zh', 'outputFormat': 'json'}
        with open(out_path, 'w', encoding='utf-8') as f2:
            with open(file_path, 'r', encoding='utf-8')as f:
                for line in f:
                    data = json.loads(line)
                    try:
                        words, tags, dependencies = corenlp_sent(nlp, data['fact_cut'].replace(' ', ''), props)
                    except Exception:
                        f2.write(json.dumps({'number_cols': [], 'dependencies': []},
                                            ensure_ascii=False))
                        f2.write('\n')
                        continue

                    # save numbers and units as cols
                    number_cols = {}
                    for i in range(len(tags) - 1):
                        if tags[i][1] == u'CD' and tags[i + 1][1] == u'M':
                            number_cols[i] = tags[i][0], tags[i + 1][0]

                    dependency_list = []
                    for each in dependencies:
                        if each[0] != u'ROOT' and each[0] != u'mark:clf':
                            first, second = words[each[1] - 1], words[each[2] - 1]
                            if each[1] - 1 in number_cols.keys():
                                first = number_cols[each[1] - 1]
                            if each[2] - 1 in number_cols.keys():
                                second = number_cols[each[2] - 1]
                            dependency_list.append([first, second])

                    f2.write(json.dumps({'number_cols': list(number_cols.values()), 'dependencies': dependency_list},
                                        ensure_ascii=False))
                    f2.write('\n')


def slice_data(filename):
    count = 0
    i = 0
    f2 = open(f'data/{filename}_part{i}.json', 'w', encoding='utf-8')
    with open(f'data/{filename}.json', 'r', encoding='utf-8') as f:
        for line in f:
            count += 1
            f2.write(line)
            if count % 10000 == 0:
                f2.close()
                i += 1
                f2 = open(f'data/{filename}_part{i}.json', 'w', encoding='utf-8')


def filter_and_re_gen(file_path, new_path, indices):
    with open(file_path, 'r')as f:
        count = 0
        with open(new_path, 'w') as f2:
            for line in f:
                count += 1
                if count not in indices:
                    f2.write(line)


def slice_data_averagely(data,percent):
    res = {}
    for i,each in enumerate(data):
        if each not in res:
            res[each] = [i]
        else:
            res[each].append(i)

    new_data_indices=[]
    for each in res:
        new_data_indices+=random.sample(res[each],int(len(res[each])*percent)+1)
    random.shuffle(new_data_indices)
    return new_data_indices

def convert_data(percent):
    word2id = pickle.load(open(f'data/xn_data/w2id_thulac.pkl', 'rb'))
    id2w={j:i for i,j in word2id.items()}
    doc_len=15
    sent_len=100
    tfidf_path='data/tfidf2.model'
    x, law, accu, imprisonments=pickle.load(open(f'data/xn_data/train_processed_thulac_Legal_basis_large_{percent}.pkl','rb'))
    x_concrete=[[[w for w in sent if w!=word2id['BLANK']] for sent in each] for each in x]
    n_words=[[len(sent) for sent in each] for each in x_concrete]

    def ending_process(x, law, accu, imprisonments, n_words):
        content = [[[id2w[w] for w in s] for s in each] for each in x]
        space_context = [' '.join(flatten(each)) for each in content]
        n_sent = [len(i) if len(i) < doc_len else doc_len for i in n_words]
        n_words = trun_n_words(n_words, sent_len)
        n_words = align_flatten2d(n_words, doc_len, flatten=False)
        # x = lookup_index_for_sentences(x, word2id, doc_len, sent_len)

        # x=[[word2id[j] for j in i.split()] for i in x]
        TFIDF = pickle.load(open(tfidf_path, 'rb'))
        tfidf = TFIDF.transform(space_context)
        batches = list(zip(x, tfidf, law, accu, imprisonments, n_sent, n_words))

        return batches

    pickle.dump(ending_process(x, law, accu, imprisonments,n_words),open(f'data/legal_data_big_{percent}_random_wv_thulac_15-100-cut_sent-True.pkl','wb'))



if __name__ == '__main__':
    # for percent in [.2,.4,.6,.8]:
    #     batches_train = list(pickle.load(open('data/xn_data/train_processed_thulac_large.pkl', 'rb')).values())
    #     # batches_val = list(pickle.load(open('data/xn_data/valid_processed_thulac.pkl', 'rb')).values())
    #     # batches_test = list(pickle.load(open('data/xn_data/test_processed_thulac_large.pkl', 'rb')).values())
    #     indices=slice_data_averagely(batches_train[2],percent)
    #     pickle.dump(indices, open(f'data/xn_data/indices_{percent}.pkl','wb'))
    #     batches_train=np.array(list(zip(*batches_train)))[indices]
    #     pickle.dump(list(zip(*batches_train)),open(f'data/xn_data/train_processed_thulac_large_{percent}.pkl','wb'))
        # del batches_train
    # for percent in [.2, .4, .6, .8]:
    #     indices=pickle.load(open(f'data/xn_data/indices_{percent}.pkl', 'rb'))
    #     batches_train2 = pickle.load(open('data/xn_data/train_processed_thulac_Legal_basis_large.pkl', 'rb'))
    #     batches_train2=[each for i,each in enumerate(batches_train2) if i in indices]
    #     # batches_train2=np.array(list(zip(*batches_train2)))[indices]
    #     pickle.dump(list(zip(*batches_train2)),open(f'data/xn_data/train_processed_thulac_Legal_basis_large_{percent}.pkl','wb'))
        # del batches_train2
    convert_data(0.2)
    convert_data(0.4)
    convert_data(0.6)
    convert_data(0.8)

    # ============gen_col=======================================================================
    # cail_data_filter()
    # slice_data('train_cs_big')
    # slice_data('test_cs_big')
    # def allocate_files(id, prefix='train'):
    #     get_collocations(f'data/{prefix}_cs_big_part{id}.json', f'data/{prefix}_col_big_{id}.json')

    # id_pool=list(range(159))
    # del id_pool[1:10]
    # p_pool = multiprocessing.Pool(10)
    # for i in range(len(id_pool)):
    #     p_pool.apply_async(get_collocations, (f'data/train_cs_big_part{id_pool[i]}.json',f'data/train_col_big_{id_pool[i]}.json'))
    #     # del id_pool[i]
    # p_pool.close()
    # p_pool.join()
    #
    # id_pool=list(range(19))
    # p_pool = multiprocessing.Pool(10)
    # for i in range(len(id_pool)):
    #     p_pool.apply_async(get_collocations, (f'data/test_cs_big_part{id_pool[i]}.json',f'data/test_col_big_{id_pool[i]}.json'))
    #     # del id_pool[0]
    # p_pool.close()
    # p_pool.join()

    # get_collocations('data/valid_cs.json', 'data/val_col.json')
    # get_collocations('data/test_cs.json', 'data/test_col.json')
    # with open('data/train_col_big.json', 'w', encoding='utf-8') as f:
    #     for i in range(159):
    #         with open(f'data/train_col_big_{i}.json', 'r', encoding='utf-8') as f1:
    #             for line in f1:
    #                 f.write(line)
    # with open('data/test_col_big.json', 'w', encoding='utf-8') as f:
    #     for i in range(19):
    #         with open(f'data/test_col_big_{i}.json', 'r', encoding='utf-8') as f1:
    #             for line in f1:
    #                 f.write(line)
    # indices=pickle.load(open('data/xn_data/indices_filtered_big.pkl','rb'))
    # # prefix=['train','val','test']
    # prefix=['train','test']
    # for i in range(2):
    #     filter_and_re_gen(f'data/{prefix[i]}_col_big.json',f'data/{prefix[i]}_col_big_filtered.json',indices[i])

    # dicts, accu_data, law_data=read_data('train')
    # # count(accu_data)
    # law_data=convert_to_chapter(law_data)
    # x,y=cut_data(law_data)
    # y=MultiLabelBinarizer().fit_transform(y)
    # tfidf=TfidfVectorizer()
    # clf=OneVsRestClassifier(LinearSVC())
    # x=tfidf.fit_transform(x)
    # clf.fit(x,y)
    # # classifier=Pipeline([('tfidf',TfidfVectorizer()),
    # #                      ('clf',OneVsRestClassifier(LinearSVC()))])
    # # classifier.fit(x,y)
    #
    # dicts, accu_data, law_data = read_data('test')
    # law_data = convert_to_chapter(law_data)
    # x, y = cut_data(law_data)
    # y=MultiLabelBinarizer().fit_transform(y)
    # print(clf.score(tfidf.transform(x),y))

    # ===============================分桶===========================
    #
    # dicts, accu_data, law_data = read_data('train')
    # x, y,_ = cut_data(law_data)
    # law_data = convert_to_chapter(law_data)
    # _, y_,_ = cut_data(law_data)
    # y = MultiLabelBinarizer().fit_transform(y)
    # y_ = MultiLabelBinarizer().fit_transform(y_)
    # tfidf = TfidfVectorizer()
    # clf = OneVsRestClassifier(LinearSVC())
    # # clf = OneVsRestClassifier(RandomForestRegressor())
    # x = tfidf.fit_transform(x)
    # clf.fit(x, y_)
    # clf2 = OneVsRestClassifier(LinearSVC())
    # clf2.fit(sp.hstack((x,y_),format='csr'),y)
    #
    # dicts, accu_data, law_data = read_data('test')
    # x, y, _ = cut_data(law_data)
    # # law_data = convert_to_chapter(law_data)
    # # _, y_ = cut_data(law_data)
    # y = MultiLabelBinarizer().fit_transform(y)
    # # y_ = MultiLabelBinarizer().fit_transform(y_)
    # x=tfidf.transform(x)
    # print(clf2.score(sp.hstack((x,clf.predict(x)),format='csr'), y))
    # print(clf.score(x,y))
    # ===============================分桶===========================

    # -------------------------------------------------------
    # print(sorted(law_data,key=lambda x:x[1][0]))
    # # idf=count_idf(accu_data)
    # # gen_idf_file(idf)
    # # with open('data/idf2','w', encoding='utf-8') as f:
    # #     with open('data/idf','r', encoding='utf-8') as f1:
    # #         for line in f1:
    # #             f.writelines(line.replace('\t',' '))
    # jieba.analyse.set_idf_path('data/idf')
    # s=[]
    # for fact,accu in accu_data:
    #     if '盗窃' in accu:
    #         s.append(fact)
    # s='\n'.join(s)
    # a=jieba.analyse.extract_tags(s)
    # print(a)

    # ===============================法条依赖关系===========================
    # import re,math
    # with open('data/criminal_law.txt','r',encoding='utf-8') as f:
    #     law=[]
    #     for line in f:
    #         try:
    #             tmp=re.compile(r'第.*条').search(line.strip()[:8])[0]
    #             law.append(line.strip())
    #         except TypeError:
    #             law[-1]+=line.strip()
    # rela=[]
    # for i in range(len(law)):
    #     tmp=re.compile(r'依照本[法|节]第(.*?)条').findall(law[i])
    #     if len(tmp)!=0:
    #         for each in tmp:
    #             each=hanzi_to_num(each)
    #             rela.append([i+102,each])
    # print(rela)
    #
    # dicts, accu_data, law_data = read_data('train')
    # count=0
    # shot=0
    # num=0
    # for each in law_data:
    #     num+=1
    #     if len(each[1])>=2:
    #         count+=1
    #         tmp=False
    #         for a in each[1]:
    #             for b in each[1]:
    #                 if [a,b] in rela or [b,a] in rela:
    #                     tmp=True
    #         if tmp:
    #             shot+=1
    #         # if [each[1][0],each[1][1]] in rela or [each[1][1],each[1][0]] in rela:
    #         #     shot+=1
    # print(num,count,shot)
    # print(hanzi_to_num('四'))
    # print(rela)
    # ===============================法条依赖关系===========================

    # a=np.load('E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\few_shot_emb.npy')
    # a

    # ===============================
    # law_list = law_to_list('data/criminal_law.txt')
    # cut_law(law_list,cut_penalty=True)

    # ===============================
    # gen_law_relation(law_list_path='data/law_filtered.txt')
