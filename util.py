import os
import networkx as nx
import numpy as np
import random
import torch
from dataset import *
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0
        self.max_neighbor = 0


def load_data(preprocessing, run, rois, threshold, type):
    subject_list = [subject.split('.')[0][1:] for subject in os.listdir('data/connectivity/{}/{}/{}'.format(preprocessing, run, rois))]
    subject_list.sort()

    behav = DataBehavioral()
    roi = DataNodes(rois)
    connectivity = DataEdges()

    _, behav_labels = behav.get_feature()

    g_list = []
    label_dict = {}
    feat_dict = {}
    for i, subject in enumerate(subject_list):
        if 'bold' in type: roi(subject)
        _, node_labels = roi.get_feature(type)
        if 'timeseries' in type:
            if len(node_labels[0]) < 1200:
                print('TRUNCATED TIMESERIES: {}'.format(subject))
                continue
        connectivity(preprocessing, run, rois, subject)
        _, connection = connectivity.get_adjacency(100-threshold)
        n = node_labels
        l = behav_labels['Gender'][int(subject)]
        if not l in label_dict:
            mapped = len(label_dict)
            label_dict[l] = mapped
        g = nx.Graph()
        node_tags = []
        node_features = []
        n_edges = 0
        for j, node_label in enumerate(n.keys()):
            g.add_node(j)
            row = [node_labels[node_label]]
            if j in connection:
                row += [len(connection[j])]
                row += connection[j]
            else:
                row += [0]
            tmp = int(row[1]) + 2
            if tmp == len(row):
                # no node attributes
                # row = [int(w) for w in row]
                attr = None
            else:
                row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
            if type=='one_hot':
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])
            else:
                node_tags.append(row[0])

            if tmp > len(row):
                node_features.append(attr)

            n_edges += row[1]
            for k in range(2, len(row)):
                g.add_edge(j, row[k])

        if node_features != []:
            node_features = np.stack(node_features)
            node_feature_flag = True
        else:
            node_features = None
            node_feature_flag = False
        assert len(g) == len(n)

        g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    #Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        if type=='one_hot':
            g.node_features = torch.zeros(len(g.node_tags), len(tagset))
            g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1
        else:
            g.node_features = torch.zeros(len(g.node_tags), len(g.node_tags[0]))
            for i in range(len(g.node_tags)):
                for j in range(len(g.node_tags[0])):
                    g.node_features[i, j] = g.node_tags[i][j]
    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list
