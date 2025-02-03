import numpy as np
import random
import math
import time
from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from application.utils.sampling.Sampler import *
from application.utils.sampling.SamplingMethods import *
from application.utils.pickle import *

class DataSampler(object):

    def __init__(self, default_sample_num):
        super().__init__()
        self.default_sample_num = default_sample_num
        self.max_sample_num = 100000
        self.test_without_sample = False
        self.sampler = Sampler()
        self.sampling_method = OutlierBiasedDensityBasedSampling # MultiClassBlueNoiseSamplingFAISS
        self.cache_root = './cache'
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
    
    def set_cache_path(self, dataset):
        self.cache_path = os.path.join(self.cache_root, dataset)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def topSampling(self, X_feature: np.ndarray, labels: np.ndarray, cache=True):
        # 顶层采样, return sampled_id + sample_addition(额外信息，例如下挂关系)
        # 0. load cache
        if cache:
            cache_file = os.path.join(self.cache_path, 'topSampling.pkl')
            if os.path.exists(cache_file):
                cache_data = load_pickle(cache_file)
                return cache_data['sampled_ids'], cache_data['sample_addition']
            
        # 1. sample total samples
        start = time.time()
        self.sampler.set_data(X_feature, labels)
        rs_args = {'sampling_rate': min(self.max_sample_num / X_feature.shape[0], 1)}
        self.sampler.set_sampling_method(RandomSampling, **rs_args)
        all_sampled_ids = self.sampler.get_samples_idx()
        # print('all_sampled_ids', all_sampled_ids.shape, time.time() - start)

        # 2. sample top samples
        start = time.time()
        num = self.default_sample_num
        self.sampler.set_data(X_feature[all_sampled_ids], self.normalizeLabels(labels[all_sampled_ids]))
        rs_args = {'sampling_rate': min((num + 1) / all_sampled_ids.shape[0], 1)}
        self.sampler.set_sampling_method(self.sampling_method, **rs_args)
        sampled_ids = self.sampler.get_samples_idx()[:num]
        sampled_ids = all_sampled_ids[sampled_ids]
        if self.test_without_sample:
            sampled_ids = list(range(num))
        # print('sampled_ids', sampled_ids.shape, time.time() - start)

        # 3. get additions: nearest items
        start = time.time()
        sampled_ids = np.array(sampled_ids)
        hang_index = np.setdiff1d(all_sampled_ids, sampled_ids)
        # sample_addition = self.getNearestHangIndex(X_feature, sampled_ids, hang_index)
        sample_addition = hang_index
        # print('sample_addition', len(sample_addition), time.time() - start)
        # from IPython import embed; embed();

        # 4. save cache
        if cache:
            cache_data = {'sampled_ids': sampled_ids, 'sample_addition': sample_addition}
            save_pickle(cache_data, cache_file)
        return sampled_ids, sample_addition

    def getSampleRange(self, sampled_id_before, sampled_addition_before, selected):
        selected_id = sampled_id_before[selected]
        hang_items = []
        for i in selected:
            hang_items.extend(sampled_addition_before[i])
        sample_range = np.concatenate([selected_id, np.array(hang_items).astype('int')], axis=0)
        return sample_range

    def zoomSampling(self, X_feature, labels, sampled_id_before, sampled_addition_before, layout_before, embedded_before, selected, bflabels=None, cache=False):
        '''
            num: 目标采样个数
            X_feature: [num_tot X f]
            labels: [num_tot]
            sampled_id_before: [num_before], 上一层的sample id
            sampled_addition_before: [num_before], 上一层的sample的额外信息
            layout_before: [N_before], 上一层的layout
            embedded_before: [num_before X 2], 上一层sample的embedded
            selected: [N_s], zoom时选中的元素在上层的序号(0..num_before-1)
        ''' 
        # 1. get selected samples
        selected_id = sampled_id_before[selected]
        hang_items = []
        for i in selected:
            hang_items.extend(sampled_addition_before[i])
        res_num = self.default_sample_num - len(selected)

        if len(hang_items) < res_num:
            total_num = len(selected) + len(hang_items)
            start = round(math.sqrt(self.default_sample_num))
            while total_num < start * start:
                start -= 1
            start += 1 
            res_num = min(start * start, total_num) - len(selected)
        
        # 2. get zoom in samples
        if len(hang_items) == 0:
            other_sample_ids = []
        else:
            self.sampler.set_data(X_feature[hang_items], self.normalizeLabels(labels[hang_items]))
            rs_args = {'sampling_rate': min((res_num + 1) / len(hang_items), 1)}
            self.sampler.set_sampling_method(self.sampling_method, **rs_args)
            # from IPython import embed; embed()
            time1 = time.time()
            other_sample_ids = self.sampler.get_samples_idx()[:res_num]
            # print('zoomSampling', X_feature[hang_items].shape, rs_args, res_num, time.time() - time1)

        other_sample_ids = np.array(hang_items)[other_sample_ids]
        if self.test_without_sample:
            other_sample_ids = []
            gap = max(math.floor(len(hang_items) / res_num), 1)
            other_sample_ids = [hang_items[int(i * gap)] for i in range(res_num)]

        # 3. get additions: nearest items
        sampled_ids = np.concatenate([selected_id, other_sample_ids], axis=0).astype(np.int32)
        hang_index = np.setdiff1d(hang_items, other_sample_ids)
        # sample_addition = self.getNearestHangIndex(X_feature, sampled_ids, hang_index)
        sample_addition = hang_index
        # print("sample_ids", sampled_ids)
        return sampled_ids, sample_addition
    
    def zoomSampling2(self, X_feature, labels, sampled_id_before, sampled_addition_before, layout_before, embedded_before, selected, bflabels=None, limit=0.8):
        # use bflabels to get more balanced samples
        # limit: 保持比例允许限制的最小节点比例
        # 1. get selected samples
        selected_id = sampled_id_before[selected]
        label_hang_items = {}
        label_ratio = {}
        for i in selected:
            bflabel = bflabels[i]
            if bflabel not in label_hang_items:
                label_hang_items[bflabel] = []
                label_ratio[bflabel] = 0
            label_hang_items[bflabel].extend(sampled_addition_before[i])
            label_ratio[bflabel] += 1

        hang_num = 0
        for key in label_hang_items:
            hang_num += len(label_hang_items[key])
            label_ratio[key] /= len(selected)

        # 2. calculate sample total num
        cur_total = len(selected) + hang_num
        if cur_total < self.default_sample_num:
            start = round(math.sqrt(self.default_sample_num))
            while cur_total < start * start:
                start -= 1
            start += 1 
            cur_total = min(start * start, cur_total)
        else:
            cur_total = self.default_sample_num
        limit_total = min(round(cur_total * limit), cur_total)

        for key in label_hang_items:
            ratio_total = round(len(label_hang_items[key]) / label_ratio[key]) + len(selected)
            if ratio_total < limit_total:
                cur_total = limit_total
                break
            if ratio_total < cur_total:
                cur_total = ratio_total
        cur_total = max(cur_total, len(selected))
        res_num = cur_total - len(selected)
        
        label_ratio2 = {}
        for key in label_hang_items:
            if len(label_hang_items[key])==0:
                label_ratio2[key] = 1
            else:
                label_ratio2[key] = res_num * label_ratio[key] / len(label_hang_items[key])
        # print("label ratio", label_ratio)
        
        # 3. get samples based on ratio
        other_sample_ids = []
        other_res_ids = []
        if res_num >= 0:
            for key in label_hang_items:
                if len(label_hang_items[key])==0:
                    continue
                items = np.array(label_hang_items[key])
                self.sampler.set_data(X_feature[items], self.normalizeLabels(labels[items]))
                rs_args = {'sampling_rate': min(label_ratio2[key], 1)}
                self.sampler.set_sampling_method(self.sampling_method, **rs_args)
                sample_idxs = items[self.sampler.get_samples_idx()]
                res_idxs = np.setdiff1d(items, sample_idxs)
                # print("key", key, len(sample_idxs))
                other_sample_ids.extend(sample_idxs)
                other_res_ids.extend(res_idxs)
            if len(other_sample_ids) >= res_num:
                resample_ids = random.sample(other_sample_ids, res_num)
                other_res_ids.extend(np.setdiff1d(other_sample_ids, resample_ids))
                other_sample_ids = resample_ids
            else:
                addition_ids = random.sample(other_res_ids, res_num - len(other_sample_ids))
                other_sample_ids.extend(addition_ids)
                other_res_ids = np.setdiff1d(other_res_ids, addition_ids)
        other_sample_ids = np.array(other_sample_ids)
        other_res_ids = np.array(other_res_ids)
        
        # 4. get additions: nearest items
        sampled_ids = np.concatenate([selected_id, other_sample_ids], axis=0).astype(np.int32)
        hang_index = other_res_ids
        # sample_addition = self.getNearestHangIndex(X_feature, sampled_ids, hang_index)
        sample_addition = hang_index
        # from IPython import embed; embed()
        return sampled_ids, sample_addition

    def zoomSampling3(self, X_feature, labels, sampled_id_before, sampled_addition_before, layout_before, embedded_before, selected, bflabels=None, limit=0.8):
        # use bflabels to get more balanced samples
        # limit: 保持比例允许限制的最小节点比例
        # 1. get selected samples
        selected_id = sampled_id_before[selected]
        label_hang_items = {}
        label_ratio = {}
        for i in selected:
            bflabel = bflabels[i]
            if bflabel not in label_hang_items:
                label_hang_items[bflabel] = []
                label_ratio[bflabel] = 0
            label_hang_items[bflabel].extend(sampled_addition_before[i])
            label_ratio[bflabel] += 1

        hang_num = 0
        for key in label_hang_items:
            hang_num += len(label_hang_items[key])
            label_ratio[key] = 1/len(label_hang_items)

        # 2. calculate sample total num
        cur_total = len(selected) + hang_num
        if cur_total < self.default_sample_num:
            start = round(math.sqrt(self.default_sample_num))
            while cur_total < start * start:
                start -= 1
            start += 1
            cur_total = min(start * start, cur_total)
        else:
            cur_total = self.default_sample_num
        limit_total = min(round(cur_total * limit), cur_total)

        for key in label_hang_items:
            ratio_total = round(len(label_hang_items[key]) / label_ratio[key]) + len(selected)
            if ratio_total < limit_total:
                cur_total = limit_total
                break
            if ratio_total < cur_total:
                cur_total = ratio_total
        cur_total = max(cur_total, len(selected))
        res_num = cur_total - len(selected)

        label_ratio2 = {}
        for key in label_hang_items:
            if len(label_hang_items[key]) == 0:
                label_ratio2[key] = 1
            else:
                label_ratio2[key] = res_num * label_ratio[key] / len(label_hang_items[key])
        # print("label ratio", label_ratio)

        # 3. get samples based on ratio
        other_sample_ids = []
        other_res_ids = []
        if res_num >= 0:
            for key in label_hang_items:
                if len(label_hang_items[key]) == 0:
                    continue
                items = np.array(label_hang_items[key])
                self.sampler.set_data(X_feature[items], self.normalizeLabels(labels[items]))
                rs_args = {'sampling_rate': min(label_ratio2[key], 1)}
                self.sampler.set_sampling_method(self.sampling_method, **rs_args)
                sample_idxs = items[self.sampler.get_samples_idx()]
                res_idxs = np.setdiff1d(items, sample_idxs)
                # print("key", key, len(sample_idxs))
                other_sample_ids.extend(sample_idxs)
                other_res_ids.extend(res_idxs)
            if len(other_sample_ids) >= res_num:
                resample_ids = random.sample(other_sample_ids, res_num)
                other_res_ids.extend(np.setdiff1d(other_sample_ids, resample_ids))
                other_sample_ids = resample_ids
            else:
                addition_ids = random.sample(other_res_ids, res_num - len(other_sample_ids))
                other_sample_ids.extend(addition_ids)
                other_res_ids = np.setdiff1d(other_res_ids, addition_ids)
        other_sample_ids = np.array(other_sample_ids)
        other_res_ids = np.array(other_res_ids)

        # 4. get additions: nearest items
        sampled_ids = np.concatenate([selected_id, other_sample_ids], axis=0).astype(np.int32)
        hang_index = other_res_ids
        # sample_addition = self.getNearestHangIndex(X_feature, sampled_ids, hang_index)
        sample_addition = hang_index
        # from IPython import embed; embed()
        return sampled_ids, sample_addition

    def getNearestHangIndex(self, X, sampled_index, hang_index, selected=None, getNowlabels=None):


        if getNowlabels is not None:
            forest_dict = {}
            sampled_labels = getNowlabels(sampled_index)
            hang_labels = getNowlabels(hang_index)
            if selected is None:
                for i, index in enumerate(sampled_index):
                    if sampled_labels[i] not in forest_dict:
                        forest_dict[sampled_labels[i]] = AnnoyIndex(X.shape[1], 'euclidean')
                    forest_dict[sampled_labels[i]].add_item(i, X[index])
            else:
                for i in selected:
                    index = sampled_index[i]
                    if sampled_labels[i] not in forest_dict:
                        forest_dict[sampled_labels[i]] = AnnoyIndex(X.shape[1], 'euclidean')
                    forest_dict[sampled_labels[i]].add_item(i, X[index])

            for key in forest_dict:
                forest_dict[key].build(10)
            sample_addition = [[] for _ in range(sampled_index.shape[0])]
            for i, index in enumerate(hang_index):
                if hang_labels[i] in forest_dict:
                    forest = forest_dict[hang_labels[i]]
                    ret = forest.get_nns_by_vector(X[index], 1, include_distances=False)
                    sample_addition[ret[0]].append(index)
            return sample_addition

        forest = AnnoyIndex(X.shape[1], 'euclidean')
        if selected is None:
            for i, index in enumerate(sampled_index):
                forest.add_item(i, X[index])
        else:
            for i in selected:
                index = sampled_index[i]
                forest.add_item(i, X[index])

        forest.build(10)
        sample_addition = [[] for _ in range(sampled_index.shape[0])]
        for index in hang_index:
            ret = forest.get_nns_by_vector(X[index], 1, include_distances = False)
            sample_addition[ret[0]].append(index)
        return sample_addition
    
    def normalizeLabels(self, labels):
        label_set = np.unique(labels)
        label_dict = {label: i for i, label in enumerate(label_set)}
        labels = np.array([label_dict[label] for label in labels])
        return labels