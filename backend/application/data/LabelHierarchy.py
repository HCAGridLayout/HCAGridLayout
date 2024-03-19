import numpy as np
import os
import random
from scipy.spatial.distance import cdist
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import json, pickle
from queue import PriorityQueue
from application.utils.pickle import *

def add_parent(node, parent):
    node['parent'] = parent
    if 'children' in node:
        for child in node['children']:
            add_parent(child, node['id'])

def get_maps(node, maps, names):
    if 'children' in node:
        for child in node['children']:
            get_maps(child, maps, names)
    names.append(str(node['id']))
    maps[node['id']] = maps['cur']
    maps['cur'] += 1

def modify_node(node, save, maps):
    save[str(node['id'])] = {
        'id': maps[node['id']],
        'children': [],
        'parent': str(node['parent']) if node['parent'] is not None else None,
        'rgb': node['rgb'] if 'rgb' in node else None,
    }
    if 'children' in node:
        save[str(node['id'])]['children'] = [str(child['id']) for child in node['children']]

    if 'children' in node:
        for child in node['children']:
            modify_node(child, save, maps)


class LabelHierarchy(object):
    def __init__(self):
        super().__init__()

        # an example
        self.label_hierarchy = {'label_names': ['dog', 'cat', 'lion', 'animals'],
                                'hierarchy': {'animals': {'id': 3, 'children': {'dog', 'cat', 'lion'}, 'parent': None},
                                              'dog': {'id': 0, 'children': {}, 'parent': 'animals'},
                                              'cat': {'id': 1, 'children': {}, 'parent': 'animals'},
                                              'lion': {'id': 2, 'children': {}, 'parent': 'animals'}}}
        self.labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2])
        self.features = np.zeros((20, 1024))
        self.root_path = './datasets/'
        # with open(os.path.join('./static/tree_colors.pkl'),'rb') as f:
        #     self.num_colors = pickle.load(f)
        self.no_hierarchy_datasets_list = ['Animals', 'MNIST', 'food']
        self.confs_hierarchy = None
        self.multilevel_load = False
        self.multilevel_load_samples = None
        self.load_samples = None
        self.levels = 1

    def transform(self, dataset='food101'):
        tree = self._load_json(os.path.join(self.root_path, dataset, '{}_total_cluster_rgb.json'.format(dataset)))
        data = np.load(os.path.join(self.root_path, dataset, '{}_total_p.npz'.format(dataset)))
        features = data['features']
        gts = data['labels'].reshape(-1)
        labels = data['predictions']
        add_parent(tree, None)

        maps = {}
        maps['cur'] = 0
        label_names = []
        get_maps(tree, maps, label_names)

        labels = labels.reshape(-1)
        normlabels = np.array([maps[label] for label in labels])
        hierarchy = {'label_names': label_names, 'hierarchy': {}}
        save = {}
        modify_node(tree, save, maps)
        hierarchy['hierarchy'] = save
        np.save(os.path.join(self.root_path, dataset, '{}_features.npy'.format(dataset)), features)
        np.save(os.path.join(self.root_path, dataset, '{}_labels.npy'.format(dataset)), gts)
        np.save(os.path.join(self.root_path, dataset, '{}_plabels.npy'.format(dataset)), normlabels)
        # from IPython import embed; embed()
        with open(os.path.join(self.root_path, dataset, '{}.json'.format(dataset)), 'w') as f:
            json.dump(hierarchy, f)
    
    def load(self, dataset='imagenet'):
        # 数据格式自己定一下？
        # TODO fengyuan
        """load hierarchy from file
        
        download datasets from https://cloud.tsinghua.edu.cn/d/994e8156c7594878a663/ and extract to .backend/datasets/
        
        hierarchy format:
        {
            'name0': {'id': 0, children: [name1, name2, ...], 'parent': None},
            'name1': {'id': 1, children: [], 'parent': 'name0'}
            ...
        }

        Args:
            dataset (str, optional): name of dataset. Defaults to 'imagenet'.
        """
        path = os.path.join(self.root_path, dataset)
        if dataset == 'cifar100':
            # self.features = np.load(os.path.join(path, 'cifar100_features.npy'))
            # self.labels = np.load(os.path.join(path, 'cifar100_labels.npy'))
            # self.gt_labels = np.load(os.path.join(path, 'cifar100_labels_gt.npy'))
            # self.label_hierarchy = self._load_json(os.path.join(path, 'cifar_2.json'))
            # self.confs_hierarchy = load_pickle(os.path.join(path, 'cifar100_confs_h_2.pkl'))

            self.multilevel_load = True
            self.full_features = np.load(os.path.join(path, 'cifar100_features.npy'))
            self.full_labels = np.load(os.path.join(path, 'cifar100_labels.npy'))
            self.full_gt_labels = np.load(os.path.join(path, 'cifar100_labels_gt.npy'))
            self.label_hierarchy = self._load_json(os.path.join(path, 'cifar_2.json'))
            self.full_confs_hierarchy = load_pickle(os.path.join(path, 'cifar100_confs_h_2.pkl'))

            self.multilevel_load_samples = load_pickle(os.path.join(path, 'cifar100_multilevel.pkl'))
            self.levels = len(self.multilevel_load_samples)
            self.load_samples = self.multilevel_load_samples[0]['samples']
            self.features = self.full_features[self.load_samples]
            self.labels = self.full_labels[self.load_samples]
            self.gt_labels = self.full_gt_labels[self.load_samples]
            self.confs_hierarchy = {'id_map': self.full_confs_hierarchy['id_map'], 'confs': self.full_confs_hierarchy['confs'][self.load_samples]}
        elif dataset == 'imagenet1k':
            # self.features = np.load(os.path.join(path, 'imagenet1k_features.npy'))
            # self.labels = np.load(os.path.join(path, 'imagenet1k_labels.npy'))
            # self.gt_labels = np.load(os.path.join(path, 'imagenet1k_labels_gt.npy'))
            # self.label_hierarchy = self._load_json(os.path.join(path, 'imagenet1k_1.json'))
            # self.confs_hierarchy = load_pickle(os.path.join(path, 'imagenet1k_confs_h_1.pkl'))

            self.multilevel_load = True
            self.full_features = np.load(os.path.join(path, 'imagenet1k_features.npy'))
            self.full_labels = np.load(os.path.join(path, 'imagenet1k_labels.npy'))
            self.full_gt_labels = np.load(os.path.join(path, 'imagenet1k_labels_gt.npy'))
            self.label_hierarchy = self._load_json(os.path.join(path, 'imagenet1k_1.json'))
            self.full_confs_hierarchy = None
            self.full_confs_hierarchy = load_pickle(os.path.join(path, 'imagenet1k_confs_h_1.pkl'))

            self.multilevel_load_samples = load_pickle(os.path.join(path, 'imagenet1k_multilevel.pkl'))
            self.levels = len(self.multilevel_load_samples)
            self.load_samples = self.multilevel_load_samples[0]['samples']
            self.features = self.full_features[self.load_samples]
            self.labels = self.full_labels[self.load_samples]
            self.gt_labels = self.full_gt_labels[self.load_samples]
            self.confs_hierarchy = {'id_map': self.full_confs_hierarchy['id_map'], 'confs': self.full_confs_hierarchy['confs'][self.load_samples]}
        elif dataset == 'imagenet1k_animals':
            # self.features = np.load(os.path.join(path, 'imagenet1k_animals_features.npy'))
            # self.labels = np.load(os.path.join(path, 'imagenet1k_animals_labels.npy'))
            # self.gt_labels = np.load(os.path.join(path, 'imagenet1k_labels_animals_gt.npy'))
            # self.label_hierarchy = self._load_json(os.path.join(path, 'imagenet1k_animals.json'))
            # self.confs_hierarchy = load_pickle(os.path.join(path, 'imagenet1k_animals_confs_h.pkl'))

            self.multilevel_load = True
            self.full_features = np.load(os.path.join(path, 'imagenet1k_animals_features.npy'))
            self.full_labels = np.load(os.path.join(path, 'imagenet1k_animals_labels.npy'))
            self.full_gt_labels = np.load(os.path.join(path, 'imagenet1k_animals_labels_gt.npy'))
            self.label_hierarchy = self._load_json(os.path.join(path, 'imagenet1k_animals.json'))
            self.full_confs_hierarchy = None
            self.full_confs_hierarchy = load_pickle(os.path.join(path, 'imagenet1k_animals_confs_h.pkl'))

            self.multilevel_load_samples = load_pickle(os.path.join(path, 'imagenet1k_animals_multilevel.pkl'))
            self.levels = len(self.multilevel_load_samples)
            self.load_samples = self.multilevel_load_samples[0]['samples']
            self.features = self.full_features[self.load_samples]
            self.labels = self.full_labels[self.load_samples]
            self.gt_labels = self.full_gt_labels[self.load_samples]
            self.confs_hierarchy = {'id_map': self.full_confs_hierarchy['id_map'], 'confs': self.full_confs_hierarchy['confs'][self.load_samples]}
        elif dataset == 'inat2021':
            # self.features = np.load(os.path.join(path, 'inat2021_features.npy'))
            # self.labels = np.load(os.path.join(path, 'inat2021_labels.npy'))
            # self.gt_labels = np.load(os.path.join(path, 'inat2021_labels_gt.npy'))
            # self.label_hierarchy = self._load_json(os.path.join(path, 'inat2021_2.json'))
            # self.confs_hierarchy = load_pickle(os.path.join(path, 'inat2021_confs_h_2.pkl'))

            self.multilevel_load = True
            self.full_features = np.load(os.path.join(path, 'inat2021_features.npy'))
            self.full_labels = np.load(os.path.join(path, 'inat2021_labels.npy'))
            self.full_gt_labels = np.load(os.path.join(path, 'inat2021_labels_gt.npy'))
            self.label_hierarchy = self._load_json(os.path.join(path, 'inat2021_2.json'))
            self.full_confs_hierarchy = None
            self.full_confs_hierarchy = load_pickle(os.path.join(path, 'inat2021_confs_h_2.pkl'))

            self.multilevel_load_samples = load_pickle(os.path.join(path, 'inat2021_multilevel.pkl'))
            self.levels = len(self.multilevel_load_samples)
            self.load_samples = self.multilevel_load_samples[0]['samples']
            self.features = self.full_features[self.load_samples]
            self.labels = self.full_labels[self.load_samples]
            self.gt_labels = self.full_gt_labels[self.load_samples]
            self.confs_hierarchy = {'id_map': self.full_confs_hierarchy['id_map'], 'confs': self.full_confs_hierarchy['confs'][self.load_samples]}

        self.build_hierarchy()
    
    def _load_json(self, file):
        with open(file, 'r') as f:
            return json.load(f)

    def build_hierarchy(self):
        self.label_hierarchy['first_level'] = []        
        self.label_hierarchy['id2label'] = {}
        for label in self.label_hierarchy['hierarchy'].keys():
            label_node = self.label_hierarchy['hierarchy'][label]
            self.label_hierarchy['id2label'][label_node['id']] = label
            if label_node['parent'] == None:
                self.label_hierarchy['first_level'].append(label)
        
    def get_hierarchy(self):
        return self.label_hierarchy

    def get_feature(self):
        return self.features
    

    def get_path_to_root(self, label_id):
        # get path and distance from a label to root
        # add a virtual total root forif hierarchy is a forest
        path = [label_id]
        distance = 0
        while True:
            cur_id = path[-1]
            cur_label = self.label_hierarchy['id2label'][cur_id]
            if self.label_hierarchy['hierarchy'][cur_label]['parent'] == None:
                path.append(-1)
                distance += 1
                break
            else:
                parent_label = self.label_hierarchy['hierarchy'][cur_label]['parent']
                parent_id = self.label_hierarchy['hierarchy'][parent_label]['id']
                path.append(parent_id)
                distance += 1
        return path, distance
    
    def get_label_dist(self, label_id1, label_id2, path1=None, dist1=None, path2=None, dist2=None):
        if path1 is None:
            path1, dist1 = self.get_path_to_root(label_id1)
        if path2 is None:
            path2, dist2 = self.get_path_to_root(label_id2)

        ## find the first same node in path1 and path2
        maxi = min(len(path1), len(path2))
        i = 0
        while i < maxi:
            if path1[-i-1] != path2[-i-1]:
                break
            i += 1
        return dist1 + dist2 - 2 * i + 2

    def extendLoad(self, new_load, sample_range):
        new_load['level'] += 1
        level = new_load['level']
        load_samples = new_load['load_samples']

        new_samples_list = []
        for sp in load_samples[sample_range]:
            new_samples_list.append(self.multilevel_load_samples[level]['children'][sp])
        new_samples = np.concatenate(new_samples_list).astype('int')

        new_load['load_samples'] = np.concatenate((load_samples, new_samples))
        new_load['features'] = self.full_features[new_load['load_samples']]
        new_load['labels'] = self.full_labels[new_load['load_samples']]
        new_load['gt_labels'] = self.full_gt_labels[new_load['load_samples']]
        if new_load['confs'] is not None:
            new_load['confs'] = {'id_map': self.full_confs_hierarchy['id_map'], 'confs': self.full_confs_hierarchy['confs'][new_load['load_samples']]}
        new_range = np.arange(len(new_samples), dtype='int')+len(load_samples)
        new_range = np.concatenate((sample_range, new_range))
        return new_load, new_range