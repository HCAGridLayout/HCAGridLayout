from .dataSampler import DataSampler
from .LabelHierarchy import LabelHierarchy
from application.grid.gridLayout_tsne import GridLayout as GridLayout_tsne
from application.grid.gridLayout import GridLayout as GridLayout_qap
from sklearn.cluster import DBSCAN, SpectralClustering
import numpy as np
import os, time
from queue import PriorityQueue
from scipy.spatial.distance import cdist
from collections import Counter

sample_num0 = 1600

class DataCtrler(object):

    def __init__(self, sample_num=sample_num0, info_dict=None):
        super().__init__()
        if info_dict is None:
            info_dict = {}
        self.sample_num = sample_num
        self.labels = None
        self.gt_labels = None
        self.features = None
        self.confs_hierarchy = None
        self.load_samples = None

        self.use_HV = False
        if "use_HV" in info_dict:
            self.use_HV = info_dict["use_HV"]
        else:
            info_dict["use_HV"] = self.use_HV

        self.use_conf = True
        if "use_conf" in info_dict:
            self.use_conf = info_dict["use_conf"]
        else:
            info_dict["use_conf"] = self.use_conf

        self.scenario = "ans"
        if "scenario" in info_dict:
            self.scenario = info_dict["scenario"]
            if info_dict["scenario"] == 'dendroans':
                self.px = 20
                if 'px' in info_dict:
                    self.px = info_dict['px']
        else:
            info_dict["scenario"] = self.scenario

        self.select_method = "square"
        if "select_method" in info_dict:
            self.select_method = info_dict["select_method"]
        else:
            info_dict["select_method"] = self.select_method

        if "method" in info_dict and info_dict["method"] == "tsne":
            self.method = "tsne"
            self.grider = GridLayout_tsne(self)
        else:
            self.method = "qap"
            self.grider = GridLayout_qap(self)
            self.grider_qap = self.grider
            self.grider_tsne = GridLayout_tsne(self)
            info_dict["method"] = self.method

        self.sampler = DataSampler(default_sample_num=sample_num)
        self.label_hierarchy = LabelHierarchy()
        self.sample_stack = []
        self.grid_stack = []
        self.label_stack = []
        self.clear_filterlabel = True
        self.load_stack = []
        self.grider_input_stack = []

        self.scale_alpha = 1/2
        self.ambiguity_threshold = 0.8

    def preprocess(self, filename='xxx.xxx'):
        self.dataset = filename
        self.label_hierarchy.load(filename)
        self.features = self.label_hierarchy.features
        self.labels = self.label_hierarchy.labels
        self.gt_labels = self.label_hierarchy.gt_labels
        self.hierarchy = self.label_hierarchy.label_hierarchy
        self.grider.set_cache_path(dataset=filename)
        self.sampler.set_cache_path(dataset=filename)
        self.cache_path = self.sampler.cache_path
        self.confs_hierarchy = self.label_hierarchy.confs_hierarchy
        self.load_samples = self.label_hierarchy.load_samples
        if self.label_hierarchy.multilevel_load:
            self.load_stack.append({'load_samples': self.load_samples, 'features': self.features, 'labels': self.labels, "gt_labels": self.gt_labels, 'confs': self.confs_hierarchy, 'level': 0})

    def clean_stacks(self):
        self.sample_stack = []
        self.grid_stack = []
        self.label_stack = []
        if self.label_hierarchy.multilevel_load:
            self.load_stack = []
            self.load_stack.append({'load_samples': self.label_hierarchy.load_samples, 'features': self.label_hierarchy.features, 'labels': self.label_hierarchy.labels, "gt_labels": self.label_hierarchy.gt_labels, 'confs': self.label_hierarchy.confs_hierarchy, 'level': 0})

    def get_now_labels(self, samples):
        if len(samples) == 0:
            return np.array([])
        labels_map = {}
        labels = self.labels[samples]
        hierarchy = self.hierarchy

        def get_label(cur_label):
            if cur_label not in labels_map:
                if len(self.label_stack) > 0 and cur_label in self.label_stack[-1][3]:
                    labels_map[cur_label] = cur_label
                else:
                    father_label_name = hierarchy['hierarchy'][hierarchy['id2label'][cur_label]]['parent']
                    if father_label_name is None:
                        labels_map[cur_label] = cur_label
                    else:
                        father_label = hierarchy['hierarchy'][father_label_name]['id']
                        labels_map[cur_label] = get_label(father_label)
            return labels_map[cur_label]
        
        new_labels = labels.copy()
        for i, label in enumerate(labels):
            new_labels[i] = get_label(label)
        return new_labels

    def reduce_now_labels(self, labels):
        if len(labels) == 0:
            return np.array([])
        labels_map = {}
        hierarchy = self.hierarchy

        def get_label(cur_label):
            if cur_label not in labels_map:
                if len(self.label_stack) > 0 and cur_label in self.label_stack[-1][3]:
                    labels_map[cur_label] = cur_label
                else:
                    father_label_name = hierarchy['hierarchy'][hierarchy['id2label'][cur_label]]['parent']
                    if father_label_name is None:
                        labels_map[cur_label] = cur_label
                    else:
                        father_label = hierarchy['hierarchy'][father_label_name]['id']
                        labels_map[cur_label] = get_label(father_label)
            return labels_map[cur_label]

        new_labels = labels.copy()
        for i, label in enumerate(labels):
            new_labels[i] = get_label(label)
        return new_labels

    def check_can_zoom_in(self):
        if len(self.sample_stack) > 0:
            last_sample_addition = self.sample_stack[-1][1]
            num = 0
            for addition in last_sample_addition:
                num += len(addition)
            if num > 0:
                return True
        return False

    def reduce_labels(self, labels, hierarchy, level1_range=(2, 3), level2_range=(4, 40), filter_ratio=0.01, spilit_ratio=1/6, spilit_size=200, zoom_without_expand=False):

        ori_spilit_size = spilit_size
        spilit_size = min(max(spilit_size, spilit_ratio * labels.shape[0]), spilit_ratio * labels.shape[0] * 2)
        spilit_size = min(spilit_size, ori_spilit_size*2)

        # 0. calculate filter boundary
        filter_num = min(0.1*spilit_size, max(1, np.ceil(filter_ratio * labels.shape[0])))
        # filter_num = max(1, np.ceil(filter_ratio * labels.shape[0])/2)
        filter_list = [set(), set()]

        # 1. get hierarchy and label frequency
        label_frequncy = {}
        label_nodes = {}
        for i, label in enumerate(labels):
            cur_label = label
            while True:
                if cur_label not in label_frequncy:
                    label_frequncy[cur_label] = 0
                    label_nodes[cur_label] = []
                label_frequncy[cur_label] += 1
                label_nodes[cur_label].append(i)
                cur_label_name = hierarchy['hierarchy'][hierarchy['id2label'][cur_label]]['parent']
                if cur_label_name is None:
                    break
                cur_label = hierarchy['hierarchy'][cur_label_name]['id']
        
        # 2. prepare score function 
        def getLabelScore(label):
            if not label in label_frequncy:
                label_frequncy[label] = 0
            score = label_frequncy[label]
            if score == 0:
                return -1
            children_list = hierarchy['hierarchy'][hierarchy['id2label'][label]]['children'] 
            if children_list is None or len(children_list) == 0:
                return 0
            return score
        
        def getClusterScore(label):
            # return 1
            if label_frequncy[label] > filter_num:
                return 1
            return 0
        

        # 3. reduce labels level 1
        q = PriorityQueue()
        q_cluster_num = 0

        if len(self.label_stack) > 0:
            tree_cut = self.label_stack[-1][3][:]
            for label in tree_cut:
                # if label not in label_frequncy:
                #     label_frequncy[label] = 0
                q.put((-getLabelScore(label), label))
                q_cluster_num += getClusterScore(label)
            # from IPython import embed; embed()
        else:
            for label_name in hierarchy['first_level']:
                label = hierarchy['hierarchy'][label_name]['id']
                q.put((-getLabelScore(label), label))
                q_cluster_num += getClusterScore(label)

            first_labels = [0 for _ in range(len(labels))]
            for score, label in q.queue:
                if -score < 0:
                    continue
                for node in label_nodes[label]:
                    first_labels[node] = label

            while q_cluster_num < level1_range[0]:
                # from IPython import embed; embed()
                score, label = q.get()
                q_cluster_num -= getClusterScore(label)
                if -score <= 0:
                    q.put((score, label))
                    q_cluster_num += getClusterScore(label)
                    break

                append_list = []
                append_num = 0
                for child_name in hierarchy['hierarchy'][hierarchy['id2label'][label]]['children']:
                    child = hierarchy['hierarchy'][child_name]['id']
                    if child not in label_frequncy:
                        # continue
                        label_frequncy[child] = 0
                    append_list.append((-getLabelScore(child), child))
                    append_num += getClusterScore(child)
                    # print("append", append_list)

                if append_num < getClusterScore(label):
                    q.put((score, label))
                    q_cluster_num += getClusterScore(label)
                    break
            
                if q_cluster_num + append_num < level1_range[1] + (level1_range[0] - q_cluster_num):
                    for item in append_list:
                        q.put(item)
                    q_cluster_num += append_num
                    if q_cluster_num >= level1_range[0]:
                        break
                else:
                    q.put((score, label))
                    q_cluster_num += getClusterScore(label)
                    break

        level1_labels = [-1 for _ in range(len(labels))]
        for score, label in q.queue:
            if -score < 0:
                continue
            # if label_frequncy[label] <= filter_num:
            #     filter_list[0].add(label)
            for node in label_nodes[label]:
                level1_labels[node] = label
        # from IPython import embed; embed();
        print("p1")
        
        # 4. reduce labels level 2
        q2 = PriorityQueue()
        q2_cluster_num = q_cluster_num
        for score, label in q.queue:
            q2.put((score, label))
            # if -score <= 0:
            #     q2.put((score, label))
            #     continue
            # q2_cluster_num -= getClusterScore(label)
            # append_list = []
            # append_num = 0
            # for child_name in hierarchy['hierarchy'][hierarchy['id2label'][label]]['children']:
            #     child = hierarchy['hierarchy'][child_name]['id']
            #     if child not in label_frequncy:
            #         # continue
            #         label_frequncy[child] = 0
            #     append_list.append((-getLabelScore(child), child))
            #     append_num += getClusterScore(child)
            # for item in append_list:
            #     q2.put(item)
            # q2_cluster_num += append_num

        # from IPython import embed; embed()
        while q2_cluster_num < level2_range[0] or -q2.queue[0][0] > spilit_size:
            score, label = q2.get()
            q2_cluster_num -= getClusterScore(label)
            if -score <= 0: # or len(q2.queue) + len(hierarchy['hierarchy'][hierarchy['id2label'][label]]['children']) > level2_range[1]:
                q2.put((score, label))
                q2_cluster_num += getClusterScore(label)
                break

            append_list = []
            append_num = 0
            for child_name in hierarchy['hierarchy'][hierarchy['id2label'][label]]['children']:
                child = hierarchy['hierarchy'][child_name]['id']
                if child not in label_frequncy:
                    # continue
                    label_frequncy[child] = 0
                append_list.append((-getLabelScore(child), child))
                append_num += getClusterScore(child)

            if append_num < getClusterScore(label):
                q2.put((score, label))
                q2_cluster_num += getClusterScore(label)
                break
                
            if q2_cluster_num + append_num < level2_range[1] + (level2_range[0] - q2_cluster_num) or -score > spilit_size:
                for item in append_list:
                    q2.put(item)
                q2_cluster_num += append_num
                # if q2_cluster_num >= level2_range[0]:
                #     break
            else:
                q2.put((score, label))
                q2_cluster_num += getClusterScore(label)
                break
        # print("???",q2_cluster_num, level2_range[0], level2_range[1])
        print("p2", filter_list)
        # from IPython import embed; embed()

        if zoom_without_expand:
            q2 = PriorityQueue()
            q2_cluster_num = q_cluster_num
            for score, label in q.queue:
                q2.put((score, label))

        largest = None
        for score, label in q2.queue:
            if largest is None or label_frequncy[label] > label_frequncy[largest]:
                largest = label
                
        level2_labels = [0 for _ in range(len(labels))]
        tree_cut = []
        for score, label in q2.queue:
            tree_cut.append(label)
            if -score < 0:
                continue
            if (label != largest) and (label_frequncy[label] <= filter_num):
                filter_list[1].add(label)
            for node in label_nodes[label]:
                level2_labels[node] = label

        # for score, label in q.queue:
        #     if -score < 0:
        #         continue
        #     if bflabels is not None and label not in bflabels:
        #         filter_list[0].add(label)
        #         for i, label in enumerate(level1_labels):
        #             lv2_label = level2_labels[i]

        level1_labels = np.array(level1_labels)
        level2_labels = np.array(level2_labels)

        filter_list = [list(filter_list[0]), list(filter_list[1])]

        if len(self.label_stack) == 0:
            level1_labels = np.array(first_labels)
            filter_list[0] = []

        self.label_stack.append([level1_labels, level2_labels, filter_list, tree_cut])
        return level1_labels, level2_labels, filter_list
    
    # def partition(self, top_labels: np.ndarray, labels: np.ndarray, label_alpha=0.8):
    #     # partition by label hierarchy
    #     # calculate distance matrix
    #     unique_labels = {}
    #     unique_label_list = []
    #     for i in range(labels.shape[0]):
    #         if labels[i] not in unique_labels:
    #             unique_labels[labels[i]] = top_labels[i]
    #             unique_label_list.append(labels[i])
    #     unique_label_num = len(unique_label_list)
    #     dist_matrix = np.zeros((unique_label_num, unique_label_num))
    #     label_paths, label_dists = [], []
    #     for i in range(unique_label_num):
    #         path, dist = self.label_hierarchy.get_path_to_root(unique_label_list[i])
    #         label_paths.append(path)
    #         label_dists.append(dist)
    #     for i in range(unique_label_num):
    #         for j in range(i + 1, unique_label_num):
    #             dist = self.label_hierarchy.get_label_dist(unique_label_list[i], unique_label_list[j], 
    #                                                          label_paths[i], label_dists[i], label_paths[j], label_dists[j])
    #             if unique_labels[labels[i]] == unique_labels[labels[j]]:
    #                 dist = dist * label_alpha
    #             dist_matrix[i, j] = dist
    #             dist_matrix[j, i] = dist
        
    #     # partition by dbscan clustering
    #     cluster_labels = DBSCAN(eps=0.5, min_samples=2, metric='precomputed').fit_predict(dist_matrix)
    #     cluster_labels = np.array(cluster_labels) - np.min(cluster_labels)
    #     label2cluster = {}
    #     for i in range(unique_label_num):
    #         label2cluster[unique_label_list[i]] = cluster_labels[i]
    #     cluster_labels = np.array([label2cluster[label] for label in labels])
    #     return cluster_labels, labels

    def processTopSampling(self, pre_sampled_id):
        if pre_sampled_id is not None and not self.label_hierarchy.multilevel_load:
            sampled_id, sampled_addition = np.array(pre_sampled_id), np.array([])
        else:
            sampled_id, sampled_addition = self.sampler.topSampling(self.features, self.labels)
        self.sample_stack.append([sampled_id, sampled_addition])
        return self.features[sampled_id], self.labels[sampled_id], sampled_id
    
    def test(self):
        grid_asses = np.array(list(range(25)))
        labels = np.array(list(range(25)))
        top_labels = np.array(list(range(25)))
        partition = np.array(list(range(25)))
        sampled_id = np.array(list(range(25)))
        similarity_info = {
            "numpy": np.zeros((25, 25)),
            "matrix": np.ones((25, 25)).tolist(),
            "labels": list(range(25))
        }
        return grid_asses, labels, top_labels, partition, sampled_id, similarity_info

    def getTopLayout(self, pre_sampled_id=None):
        self.clean_stacks()

        if self.label_hierarchy.multilevel_load:
            new_load = {'load_samples': self.load_stack[-1]['load_samples'], 'features': self.load_stack[-1]['features'],
                        'labels': self.load_stack[-1]['labels'], 'gt_labels': self.load_stack[-1]['gt_labels'], 'confs': self.load_stack[-1]['confs'],
                        'level': self.load_stack[-1]['level']}
            self.features = new_load['features']
            self.labels = new_load['labels']
            self.gt_labels = new_load['gt_labels']
            self.confs_hierarchy = new_load['confs']
            self.load_samples = new_load['load_samples']
            self.load_stack.append(new_load)

        data, labels, sampled_id = self.processTopSampling(pre_sampled_id=pre_sampled_id)
        top_labels, labels, filter_labels = self.reduce_labels(labels, self.hierarchy)
        gt_labels = self.reduce_now_labels(self.gt_labels[sampled_id])

        old_sampled_id = self.sample_stack[-1][0]
        old_hang_id = self.sample_stack[-1][1]
        self.sample_stack[-1][1] = self.sampler.getNearestHangIndex(self.features, old_sampled_id, old_hang_id, getNowlabels=self.get_now_labels)

        self.grider_input_stack.append([data.copy(), labels.copy(), sampled_id.copy(), None, None, top_labels.copy(), [filter_labels[0].copy(), filter_labels[1].copy()], self.confs_hierarchy])
        X_embedded, grid_asses, grid_size, partition, part_info, top_part, confusion = self.grider.fit(data, labels, sampled_id, self.hierarchy, top_labels=top_labels, filter_labels=filter_labels, confs_hierarchy=self.confs_hierarchy, scale=self.scale_alpha, shres=self.ambiguity_threshold)
        similarity_info = self.get_label_similar(data, labels)
        self.grid_stack.append([X_embedded, grid_asses, partition, similarity_info, part_info, data, top_part, None, confusion])
        return grid_asses, labels, top_labels, gt_labels, partition, sampled_id, similarity_info, data, top_part, None, confusion

    def processZoomSampling(self, selected, usebflabels=True, new_range=None, pre_sampled_id=None):
        data = self.features
        labels = self.labels
        bflabels = None
        if usebflabels:
            bflabels = self.label_stack[-1][1]
        # 从当前层向下zoom

        old_sampled_id = self.sample_stack[-1][0]
        old_sampled_addition = self.sample_stack[-1][1]
        if new_range is not None:
            old_sampled_addition = self.sampler.getNearestHangIndex(data, old_sampled_id, np.setdiff1d(new_range, old_sampled_id), selected, self.get_now_labels)

        if pre_sampled_id is not None and not self.label_hierarchy.multilevel_load:
            sampled_id, sampled_addition = np.array(pre_sampled_id), np.array([])
            sampled_id = np.concatenate([old_sampled_id[selected], np.setdiff1d(sampled_id, old_sampled_id[selected])])
        else:

            sampled_id, sampled_addition = self.sampler.zoomSampling2(data, labels,
                                                                 old_sampled_id, old_sampled_addition,
                                                                 self.grid_stack[-1][1], self.grid_stack[-1][0],
                                                                 selected, bflabels, limit=1)
        # sampled_id, sampled_addition = self.sampler.zoomSampling(data, labels,
        #                                                          old_sampled_id, old_sampled_addition,
        #                                                          self.grid_stack[-1][1], self.grid_stack[-1][0],
        #                                                          selected)
        
        # import pickle
        # tmp_data = {"sampled_id": sampled_id, "sampled_addition": sampled_addition}
        # with open('zoomsample2.pkl', 'wb') as file:
        #     pickle.dump(tmp_data, file)
        # with open('zoomsample2.pkl', 'rb') as file:
        #     tmp_data = pickle.load(file)
        # sampled_id = tmp_data["sampled_id"]
        # sampled_addition = tmp_data["sampled_addition"]
        
        self.sample_stack.append([sampled_id, sampled_addition])
        return data[sampled_id], labels[sampled_id], sampled_id

    def gridZoomIn(self, selected, pre_sampled_id=None, zoom_without_expand=False):
        time1 = time.time()

        new_range = None
        if self.label_hierarchy.multilevel_load:
            new_load = {'load_samples': self.load_stack[-1]['load_samples'], 'features': self.load_stack[-1]['features'],
                        'labels': self.load_stack[-1]['labels'], 'gt_labels': self.load_stack[-1]['gt_labels'], 'confs': self.load_stack[-1]['confs'],
                        'level': self.load_stack[-1]['level']}
            sample_range = self.sampler.getSampleRange(self.sample_stack[-1][0], self.sample_stack[-1][1], selected)
            while len(sample_range) < max(self.sample_num, min(self.sample_num*2, 10000)) and new_load['level'] < self.label_hierarchy.levels-1:
                new_load, new_range = self.label_hierarchy.extendLoad(new_load, sample_range)
                sample_range = new_range
            self.features = new_load['features']
            self.labels = new_load['labels']
            self.gt_labels = new_load['gt_labels']
            self.confs_hierarchy = new_load['confs']
            self.load_samples = new_load['load_samples']
            self.load_stack.append(new_load)

        data, labels, sampled_id = self.processZoomSampling(selected, new_range=new_range, pre_sampled_id=pre_sampled_id)
        time2 = time.time()
        print("zoom sampling time: ", time2 - time1)
        top_labels, labels, filter_labels = self.reduce_labels(labels, self.hierarchy, zoom_without_expand=zoom_without_expand)
        gt_labels = self.reduce_now_labels(self.gt_labels[sampled_id])

        old_sampled_id = self.sample_stack[-1][0]
        old_hang_id = self.sample_stack[-1][1]
        self.sample_stack[-1][1] = self.sampler.getNearestHangIndex(self.features, old_sampled_id, old_hang_id, getNowlabels=self.get_now_labels)

        selected_now = list(range(len(selected)))
        info_before = {"selected": selected_now,
                       "selected_bf": selected,
                       "X_embedded": self.grid_stack[-1][0],
                       "grid_asses": self.grid_stack[-1][1],
                       "partition_info_bf": self.grid_stack[-1][4],
                       "sampled_id": self.sample_stack[-2][0]}

        self.grider_input_stack.append([data.copy(), labels.copy(), sampled_id.copy(), None, info_before, top_labels.copy(), [filter_labels[0].copy(), filter_labels[1].copy()], self.confs_hierarchy])
        X_embedded, grid_asses, grid_size, partition, part_info, top_part, confusion = self.grider.fit(data, labels, sampled_id, self.hierarchy, info_before, top_labels=top_labels, filter_labels=filter_labels, confs_hierarchy=self.confs_hierarchy, scale=self.scale_alpha, shres=self.ambiguity_threshold)
        
        similarity_info = self.get_label_similar(data, labels)
        self.grid_stack.append([X_embedded, grid_asses, partition, similarity_info, part_info, data, top_part, info_before, confusion])
        return grid_asses, labels, top_labels, gt_labels, partition, sampled_id, similarity_info, data, top_part, info_before, confusion

    def gridZoomOut(self):
        if len(self.grid_stack) > 1:
            self.grid_stack.pop()
            self.sample_stack.pop()
            self.label_stack.pop()
            if self.label_hierarchy.multilevel_load:
                self.load_stack.pop()
                self.features = self.load_stack[-1]['features']
                self.labels = self.load_stack[-1]['labels']
                self.gt_labels = self.load_stack[-1]['gt_labels']
                self.confs_hierarchy = self.load_stack[-1]['confs']
                self.load_samples = self.load_stack[-1]['load_samples']
            self.grider_input_stack.pop()

        grid_asses = self.grid_stack[-1][1]
        grid_size = round(np.sqrt(grid_asses.shape[0]))
        partition = self.grid_stack[-1][2]
        sampled_id = self.sample_stack[-1][0]
        labels = self.label_stack[-1][1]
        top_labels = self.label_stack[-1][0]
        gt_labels = self.reduce_now_labels(self.gt_labels[sampled_id])
        similarity_info = self.grid_stack[-1][3]
        feature = self.grid_stack[-1][5]
        top_part = self.grid_stack[-1][6]
        info_before = self.grid_stack[-1][7]
        confusion = self.grid_stack[-1][8]
        return grid_asses, labels, top_labels, gt_labels, partition, sampled_id, similarity_info, feature, top_part, info_before, confusion

    def gridRelayout(self, scale_alpha=1/2, ambiguity_threshold=0.8):
        self.scale_alpha = scale_alpha
        self.ambiguity_threshold = ambiguity_threshold

        input = self.grider_input_stack[-1]
        X_embedded, grid_asses, grid_size, partition, part_info, top_part, confusion = self.grider.fit(input[0], input[1], input[2], input[3], input[4], top_labels=input[5], filter_labels=input[6], confs_hierarchy=input[7], scale=self.scale_alpha, shres=self.ambiguity_threshold)

        self.grid_stack[-1][0] = X_embedded
        self.grid_stack[-1][1] = grid_asses
        grid_size = round(np.sqrt(grid_asses.shape[0]))
        self.grid_stack[-1][2] = partition
        similarity_info = self.grid_stack[-1][3]
        self.grid_stack[-1][4] = part_info
        feature = self.grid_stack[-1][5]
        self.grid_stack[-1][6] = top_part
        info_before = self.grid_stack[-1][7]
        confusion = self.grid_stack[-1][8]
        sampled_id = self.sample_stack[-1][0]
        labels = self.label_stack[-1][1]
        top_labels = self.label_stack[-1][0]
        gt_labels = self.reduce_now_labels(self.gt_labels[sampled_id])

        return grid_asses, labels, top_labels, gt_labels, partition, sampled_id, similarity_info, feature, top_part, info_before, confusion

    
    def get_label_similar(self, data, labels):
        unique_labels = np.sort(np.unique(labels))
        avg_feat = np.array([np.mean(data[labels == label], axis=0) for label in unique_labels])
        dist_matrix = cdist(avg_feat, avg_feat, metric='cosine')
        # from IPython import embed; embed()
        return {
            "numpy": 1 - dist_matrix,
            "matrix": dist_matrix.tolist(), 
            "labels": unique_labels.tolist()
        }
    
    def clear_filter_labels(self, data, sampled_id, top_labels, labels, filter_labels, selected=None):
        idx1 = np.where(np.isin(top_labels, filter_labels[0]))[0]
        idx2 = np.where(np.isin(labels, filter_labels[1]))[0]
        idx_set = set(idx1) | set(idx2)
        res_idx_set = set(range(data.shape[0])) - idx_set
        final_idx = np.array(list(res_idx_set))
        data = data[final_idx]
        sampled_id = sampled_id[final_idx]
        top_labels = top_labels[final_idx]
        labels = labels[final_idx]
        self.sample_stack[-1][0] = sampled_id
        additions = []
        for idx in final_idx:
            additions.append(self.sample_stack[-1][1][idx])
        self.sample_stack[-1][1] = additions
        self.label_stack[-1][0] = top_labels
        self.label_stack[-1][1] = labels
        self.label_stack[-1][2] = [[], []]
        _selected = []
        if selected is not None:
            for i in range(len(selected)):
                if i in res_idx_set:
                    _selected.append(selected[i])
        # from IPython import embed; embed()
        return data, sampled_id, top_labels, labels, [[], []], _selected


    
# dataCtrler = DataCtrler()
# dataCtrler.preprocess('MNIST')
# print(dataCtrler.getTopLayout())
# print(dataCtrler.sample_stack[-1])
# print(dataCtrler.grid_stack[-1])