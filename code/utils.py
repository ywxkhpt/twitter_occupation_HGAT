import numpy as np
import scipy.sparse as sp
from random import shuffle
import torch
from tqdm import tqdm
import os


def load_data(path="../data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    features_block = False
    
    MULTI_LABEL = False
    
    if dataset == 'dblp':
        type_list = ['A', 'P', 'T', 'C']
        type_have_label = 'A'
    else:
        type_list = ['text', 'topic', 'entity']
        # type_list = ['text', 'topic', 'cate']
        type_have_label = 'text'

    features_list = []
    idx_map_list = []
    idx2type = {t: set() for t in type_list}

    for type_name in type_list:
        print('Loading {} content...'.format(type_name))
        print(path)
        print(dataset)
        print(type_name)
        indexes, features, labels = [], [], []
        with open("{}{}.content.{}".format(path, dataset, type_name)) as f:
            for line in tqdm(f):
                cache = line.strip().split('\t')
                indexes.append(np.array(cache[0], dtype=int))
                # if type_name == "entity":
                #     print("cache[0]",np.array(cache[0], dtype=int).size)
                features.append(np.array(cache[1:-1], dtype=np.float32))
                labels.append(np.array([cache[-1]], dtype=str) )
            # idx_features_labels = np.stack(data)
            # idx_features_labels = np.genfromtxt("{}{}.content.{}".format(path, dataset, type_name),
            #                                      delimiter="\t", dtype=np.dtype(str))
            # features = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
            features = np.stack(features)
            features = normalize(features)
            if not features_block:
                features = torch.FloatTensor(np.array(features))
                features = dense_tensor_to_sparse(features)

            features_list.append(features)
        if type_name == type_have_label:
            labels = np.stack(labels)
            if not MULTI_LABEL:
                labels = encode_onehot(labels)
            else:
                labels = multi_label(labels)
            Labels = torch.LongTensor(labels)
            print("label matrix shape: {}".format(Labels.shape))

        # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx = np.stack(indexes)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
        print('done.')

    # feature补零
    len_list = [len(idx2type[t]) for t in type_list]
    type2len = {t: len(idx2type[t]) for t in type_list}
    len_all = sum(len_list)
    if features_block:
        flen = [i.shape[1] for i in features_list]
        features = sp.lil_matrix(np.zeros((len_all, sum(flen))), dtype=np.float32)
        bias = 0
        for i_l in range(len(len_list)):
            features[bias:bias+len_list[i_l], :flen[i_l]] = features_list[i_l]
            features_list[i_l] = features[bias:bias+len_list[i_l], :]
            bias += len_list[i_l]
        for fi in range(len(features_list)):
            features_list[fi] = torch.FloatTensor(np.array(features_list[fi].todense()))
            features_list[fi] = dense_tensor_to_sparse(features_list[fi])

    print('Building graph...')
    # adj_list = [[None, None], [None, None]]
    adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
    # build graph
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            t1, t2 = type_list[i1], type_list[i2]
            if i1 == i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(type2len[t1], type2len[t2]),
                                        dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]),
                                        dtype=np.float32)
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()


            elif i1 < i2:
                edges = []
                for edge in edges_unordered:
                    if (edge[0] in idx2type[t1] and edge[1] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[0]), idx_map_list[i2].get(edge[1])])
                    elif (edge[1] in idx2type[t1] and edge[0] in idx2type[t2]):
                        edges.append([idx_map_list[i1].get(edge[1]), idx_map_list[i2].get(edge[0])])
                edges = np.array(edges)
                if len(edges) > 0:
                    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                        shape=(type2len[t1], type2len[t2]),
                                        dtype=np.float32)
                else:
                    adj = sp.coo_matrix((type2len[t1], type2len[t2]),
                                        dtype=np.float32)

                ####!!!! 这里有个迷之缩进   ！！！#####很可能别处也错了

                adj_all[
                    sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                    sum(len_list[:i2]): sum(len_list[:i2 + 1])] = adj.tolil()
                adj_all[
                    sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                    sum(len_list[:i1]): sum(len_list[:i1 + 1])] = adj.T.tolil()

            else:
                pass

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))

    for i1 in range(len(type_list)):
        for i2 in range(len(type_list)):
            adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                        sum(len_list[:i2]): sum(len_list[:i2 + 1])]
            )

    print("Num of edges: {}".format(len( adj_all.nonzero()[0] )))
    idx_train, idx_val, idx_test = load_divide_idx(path, idx_map_list[0])
    return adj_list, features_list, Labels, idx_train, idx_val, idx_test, idx_map_list[0]


def multi_label(labels):
    def myfunction(x):
        return list(map(int, x[0].split()))
    return np.apply_along_axis(myfunction, axis=1, arr=labels)


def encode_onehot(labels):
    classes = set(labels.T[0])
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels.T[0])),
                             dtype=np.int32)
    return labels_onehot

def load_divide_idx(path, idx_map):
    idx_train = []
    idx_val = []
    idx_test = []
    with open(path+'train.map', 'r') as f:
        for line in f:
            idx_train.append( idx_map.get(int(line.strip('\n'))) )
    with open(path+'vali.map', 'r') as f:
        for line in f:
            idx_val.append( idx_map.get(int(line.strip('\n'))) )
    with open(path+'test.map', 'r') as f:
        for line in f:
            idx_test.append( idx_map.get(int(line.strip('\n'))) )

    shuffle(idx_val)
    # idx_val = idx_val[:80]

    print(len(idx_train), len(idx_val), len(idx_test))
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test


def resample(train, val, test : torch.LongTensor, path, idx_map, rewrite=True):
    if os.path.exists(path+'train_inductive.map'):
        rewrite = False
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = []
        for file in filenames:
            with open(path+file+'_inductive.map', 'r') as f:
                cache = []
                for line in f:
                    cache.append(idx_map.get(int(line)))
            ans.append(torch.LongTensor(cache))
        return ans

    idx_train = train
    idx_test = val
    cache = list(test.numpy())
    shuffle(cache)
    idx_val = cache[: idx_train.shape[0]]
    idx_unlabeled = cache[idx_train.shape[0]: ]
    idx_val = torch.LongTensor(idx_val)
    idx_unlabeled = torch.LongTensor(idx_unlabeled)

    print("\n\ttrain: ", idx_train.shape[0],
          "\n\tunlabeled: ", idx_unlabeled.shape[0],
          "\n\tvali: ", idx_val.shape[0],
          "\n\ttest: ", idx_test.shape[0])
    if rewrite:
        idx_map_reverse = dict(map(lambda t: (t[1], t[0]), idx_map.items()))
        filenames = ['train', 'unlabeled', 'vali', 'test']
        ans = [idx_train, idx_unlabeled, idx_val, idx_test]
        for i in range(4):
            with open(path+filenames[i]+'_inductive.map', 'w') as f:
                f.write("\n".join(map(str, map(idx_map_reverse.get, ans[i].numpy()))))

    return idx_train, idx_unlabeled, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)




def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if len(sparse_mx.nonzero()[0]) == 0:
        # 空矩阵
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor( sp.coo.coo_matrix(dense_mx) )


def load_data_ori(path="../data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    include_topic = True


    features_list = []
    idx_map_list = []
    idx2type = {'text': set(), 'entity': set()}
    type_list = ['text', 'entity']
    labels = []
    if include_topic:
        idx2type['topic'] = set()
        type_list.append('topic')
    for type_name in type_list:
        print('Loading {} content...'.format(type_name))
        idx_features_labels = np.genfromtxt("{}{}.content.{}".format(path, dataset, type_name),
                                            dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = normalize(features)
        features_list.append(features)

        # if type_name == 'text':
            # labels = encode_onehot(idx_features_labels[:, -1])
            # labels = torch.LongTensor(np.where(labels)[1])
        labels.append(idx_features_labels[:, -1])

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        for i in idx:
            idx2type[type_name].add(i)
        idx_map = {j: i for i, j in enumerate(idx)}
        idx_map_list.append(idx_map)
        print('done.')

    labels = np.concatenate(labels)
    labels = encode_onehot(labels)
    labels = torch.LongTensor(np.where(labels)[1])

    len_text = len(idx2type['text'])
    len_entity = len(idx2type['entity'])
    len_topic = len(idx2type['topic'])
    len_all = len_text + len_entity + len_topic
    len_list = [len_text, len_entity]
    if include_topic:
        len_list.append(len_topic)

    print('Building features...')


    flen = [i.shape[1] for i in features_list]
    features = sp.lil_matrix( np.zeros((len_all, sum(flen))), dtype=np.float32)
    features[:len_text, :flen[0]] = features_list[0]
    features[len_text:len_text+len_entity, flen[0]:flen[0]+flen[1]] = features_list[1]
    if include_topic:
        features[len_text+len_entity:, flen[0]+flen[1]:] = features_list[2]
    features =  torch.FloatTensor(np.array(features.todense()))
    features = dense_tensor_to_sparse(features)

    print('Building graph...')
    # build graph
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)

    adj_all = sp.lil_matrix( np.zeros((len_all, len_all)), dtype=np.float32)

    # edge type is 'text - entity'
    edges = []
    for edge in edges_unordered:
        if (edge[0] in idx2type['text'] and edge[1] in idx2type['entity']):
            edges.append([ idx_map_list[0].get(edge[0]), idx_map_list[1].get(edge[1]) ])
        elif (edge[1] in idx2type['text'] and edge[0] in idx2type['entity']):
            edges.append([ idx_map_list[0].get(edge[1]), idx_map_list[1].get(edge[0]) ])
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones( edges.shape[0] ), (edges[:, 0], edges[:, 1])),
                            shape=(len_text, len_entity),
                            dtype=np.float32).tolil()
    adj_all[:len_text, len_text:len_text+len_entity] = adj.tolil()
    adj_all[len_text:len_text+len_entity, :len_text] = adj.T.tolil()

    # edge type is 'entity - entity'
    edges = []
    for edge in edges_unordered:
        if (edge[0] in idx2type['entity'] and edge[1] in idx2type['entity']):
            edges.append([ idx_map_list[1].get(edge[0]), idx_map_list[1].get(edge[1]) ])
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones( edges.shape[0] ), (edges[:, 0], edges[:, 1])),
                            shape=(len_entity, len_entity),
                            dtype=np.float32)
    adj_all[len_text:len_text+len_entity, len_text:len_text+len_entity] = adj.tolil()

    # edge type is 'text - text'  单位阵
    edges = [ [idx_map_list[0].get(i), idx_map_list[0].get(i)] for i in idx2type['text'] ]
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones( edges.shape[0] ), (edges[:, 0], edges[:, 1])),
                            shape=(len_text, len_text),
                            dtype=np.float32)
    adj_all[:len_text, :len_text] = adj.tolil()

    if include_topic:
        # edge type is 'text - topic'
        edges = []
        for edge in edges_unordered:
            if (edge[0] in idx2type['text'] and edge[1] in idx2type['topic']):
                edges.append([ idx_map_list[0].get(edge[0]), idx_map_list[2].get(edge[1]) ])
            elif (edge[1] in idx2type['text'] and edge[0] in idx2type['topic']):
                edges.append([ idx_map_list[0].get(edge[1]), idx_map_list[2].get(edge[0]) ])
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones( edges.shape[0] ), (edges[:, 0], edges[:, 1])),
                                shape=(len_text, len_topic),
                                dtype=np.float32)
        adj_all[:len_text, len_text+len_entity:] = adj.tolil()
        adj_all[len_text+len_entity:, :len_text] = adj.T.tolil()

         # edge type is 'topic - topic'  单位阵
        edges = [ [idx_map_list[2].get(i), idx_map_list[2].get(i)] for i in idx2type['topic'] ]
        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones( edges.shape[0] ), (edges[:, 0], edges[:, 1])),
                                shape=(len_topic, len_topic),
                                dtype=np.float32)
        adj_all[len_text+len_entity:, len_text+len_entity:] = adj.tolil()

    adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
    adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))
    adj_all = sparse_mx_to_torch_sparse_tensor(adj_all)

    idx_train, idx_val, idx_test = load_divide_idx(path, idx_map_list[0])
    return adj_all, features, labels, idx_train, idx_val, idx_test, len_list, flen

if __name__ == "__main__":
    load_data(path="../data/agnews/", dataset="agnews")