# example data for testing
import numpy as np
import base64, io
from PIL import Image
from .colors import MyColorMap
import random, os, math
from .dataCtrler import DataCtrler
from copy import deepcopy
from application.utils.pickle import *
import time

class Port(object):
    def __init__(self, sample_num=-1, info_dict=None) -> None:
        if info_dict is None:
            info_dict = {}
        if sample_num<=0:
            self.data_ctrler = DataCtrler(info_dict=info_dict)
        else:
            self.data_ctrler = DataCtrler(sample_num, info_dict=info_dict)
        self.label_hierarchy = None
        self.color_map = MyColorMap()
        self.cur_idx = 0
        self.hierarchy = {}
        self.stack = [-1]
        self.color_stack = [None]
        self.use_image = False

        self.save_port_for_eval = False
        self.port_cache = "./port_cache/"
        self.port_path = None
        self.save_id = 0

        self.sample_num = sample_num
        self.info_dict = info_dict

    def update_setting(self, sample_num=None, info_dict=None):
        if sample_num is not None:
            self.sample_num = sample_num
        self.info_dict.update(info_dict)
        if self.sample_num <= 0:
            self.data_ctrler = DataCtrler(info_dict=self.info_dict)
        else:
            self.data_ctrler = DataCtrler(self.sample_num, info_dict=self.info_dict)

    def load_dataset(self, dataset='MNIST'):
        self.data_set = dataset
        self.data_ctrler.preprocess(dataset)
        self.label_hierarchy = self.data_ctrler.label_hierarchy
        image_path = os.path.join("./datasets/", dataset, "{}_images.npz".format(dataset))
        folder_path = os.path.join("./datasets/", dataset, "{}_images".format(dataset))
        self.use_image = False
        if os.path.exists(image_path):
            image_data = np.load(image_path)
            self.images = image_data["images"]
            self.use_image = True
        elif os.path.exists(folder_path):
            self.images = folder_path
            self.use_image = True

        print("use image", self.use_image)

    def package_gridlayout(self, parent: int, index: int, grid: np.ndarray, labels: np.ndarray, top_labels: np.ndarray, gt_labels: np.ndarray, part_labels: np.array, samples: np.ndarray, similar: dict, feature, top_part, info_before, confusion, zoomout=False):
        gridlayout = {}
        gridlayout["parent"] = parent
        gridlayout["index"] = index
        gridlayout["grid"] = grid.tolist()
        gridlayout["top_labels"] = top_labels.tolist()
        gridlayout["labels"] = labels.tolist()   # predict label in treecut
        gridlayout["gt_labels"] = gt_labels.tolist()   # gt label in treecut
        gridlayout["bottom_labels"] = {"labels": self.data_ctrler.labels[samples].tolist(), "gt_labels": self.data_ctrler.gt_labels[samples].tolist()}
        gridlayout["part_labels"] = part_labels
        size = round(math.sqrt(len(gridlayout["grid"])))
        grid_info = {
            "size": size,
            "grid": grid,
            "labels": labels,
            "similarity": similar['numpy'],
        }
        gridlayout["label_names"], gridlayout["colors"], gridlayout["pcolors"], gridlayout["color_ids"] = self.get_color(labels, top_labels, part_labels, zoomout, grid_info)
        gridlayout["size"] = [size, size]
        gridlayout["sample_ids"] = samples.tolist()
        gridlayout["feature"] = feature
        gridlayout["top_part"] = top_part
        gridlayout["info_before"] = info_before
        gridlayout["confusion"] = confusion
        self.hierarchy[gridlayout["index"]] = gridlayout
        gridlayout["similarity"] = {
            "matrix": similar['matrix'],
            "labels": similar['labels']
        }
        return gridlayout

    def get_color(self, labels, top_labels, part_labels, zoomout=False, grid_info=None) -> list:
        start = time.time()
        # return np.unique(labels).tolist(), self.color_map.colorSet(list(range(np.max(labels)+1)))
        if zoomout:
            records = self.color_stack[-1]
        else:
            # history = None 
            # if self.stack[-1] > 0:
            #     history = deepcopy(self.color_stack[-1])
            history = self.color_stack[-1]
            # records = self.color_map.hierarchyColorSet(top_labels, labels, self.label_hierarchy, history)
            records = self.color_map.hierarchyColorSet2(top_labels, labels, part_labels, history, self.label_hierarchy.label_hierarchy, grid_info)

            ulabels = np.unique(labels).tolist()
            if history is not None:
                for key in records:
                    if key not in ulabels:
                        continue
                    if key in history:
                        records[key]["parent_color"] = history[key]["color"]

            self.color_stack.append(records)
        # from IPython import embed; embed()
        colornames = {}
        colormap = {}
        pcolormap = {}
        colorids = {}
        all_single = False
        ulabels = np.unique(labels).tolist()
        if len(ulabels) == len(records['single']):
            all_single = True
        for label in ulabels:
            label = int(label)
            if label not in colornames:
                colornames[label] = self.label_hierarchy.label_hierarchy['id2label'][label]
                colormap[label] = records[label]['color']
                pcolormap[label] = records[label]['parent_color']
                colorids[label] = records[label]['color_id']
                if label in records['single'] and not all_single:
                    colorids[label] = records[label]['color_id'] + '-0'
        print('color assignment time: ', time.time()-start)
        return self.label_hierarchy.label_hierarchy['id2label'], colormap, pcolormap, colorids

    def solve_gray_img(self, img):
        tp = 255 - img
        tp1 = tp.copy()
        tp2 = tp.copy()
        return np.array([tp, tp1, tp2]).reshape((3,28,28)).transpose(1,2,0)

    def get_image_instances(self, node_id, chosen_names=None) -> dict:
        assert self.use_image
        grid_info = self.hierarchy[node_id]
        base64Imgs = []
        if chosen_names is None:
            chosen_names = list(range(len(grid_info["sample_ids"])))
        for name in chosen_names:
            sample_id = grid_info["sample_ids"][name]
            if self.data_ctrler.label_hierarchy.multilevel_load:
                sample_id = self.data_ctrler.load_samples[sample_id]
            if isinstance(self.images, str):
                image_path = os.path.join(self.images, str(sample_id)+".jpeg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.images, str(sample_id)+".png")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.images, str(sample_id)+".jpg")
                jpeg_data = open(image_path, "rb").read()
                base64Imgs.append(base64.b64encode(jpeg_data).decode())
            else:
                # image = self.solve_gray_img(self.images[sample_id])
                image = self.images[sample_id]
                image = Image.fromarray(np.uint8(image)).convert('RGB')
                output = io.BytesIO()
                image.save(output, format="JPEG")
                base64Imgs.append(base64.b64encode(output.getvalue()).decode())
        return base64Imgs

    def get_image_ratio(self, sample_ids):
        ret = []
        for sample_id in sample_ids:
            if not self.use_image:
                ret.append([1.0, 1.0])
                continue
            ratio = [1.0, 1.0]
            if self.data_ctrler.label_hierarchy.multilevel_load:
                sample_id = self.data_ctrler.load_samples[sample_id]
            if isinstance(self.images, str):
                image_path = os.path.join(self.images, str(sample_id) + ".jpeg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.images, str(sample_id) + ".png")
                    if not os.path.exists(image_path):
                        image_path = os.path.join(self.images, str(sample_id) + ".jpg")
                img = Image.open(image_path)
                width, height = img.size
            else:
                image = self.images[sample_id]
                height, width = np.uint8(image).shape[0], np.uint8(image).shape[1]
            if height > width:
                ratio[0] = width / height
            else:
                ratio[1] = height / width
            ret.append(ratio)
        return ret

    def top_gridlayout(self, pre_sampled_id=None) -> dict:
        self.stack = [-1]
        self.color_stack = [None]
        grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion = self.data_ctrler.getTopLayout(pre_sampled_id=pre_sampled_id)
        self.stack.append(0)        
        return self.package_gridlayout(-1, 0, grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion)
    
    def layer_gridlayout(self, node_id, samples, pre_sampled_id=None, zoom_without_expand=False, zoom_balance=False) -> dict:
        self.cur_idx += 1
        grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion = self.data_ctrler.gridZoomIn(samples, pre_sampled_id=pre_sampled_id, zoom_without_expand=zoom_without_expand, zoom_balance=zoom_balance)
        self.stack.append(self.cur_idx)
        return self.package_gridlayout(node_id, self.cur_idx, grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion)
    
    def test(self) -> dict:
        grid, labels, top_labels, part_labels, samples, similar = self.data_ctrler.test()
        return self.package_gridlayout(-1, 0, grid, labels, top_labels, part_labels, samples, similar)
    
    def zoom_out_gridlayout(self, node_id) -> dict:
        cur_id = self.stack.pop()
        self.color_stack.pop()
        assert cur_id == node_id and node_id > 0
        grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion = self.data_ctrler.gridZoomOut()
        return self.package_gridlayout(self.stack[-2], self.stack[-1], grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion, True)

    def re_gridlayout(self, scale_alpha=1/2, ambiguity_threshold=0.8) -> dict:
        grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion = self.data_ctrler.gridRelayout(scale_alpha, ambiguity_threshold)
        return self.package_gridlayout(self.stack[-2], self.stack[-1], grid, labels, top_labels, gt_labels, part_labels, samples, similar, feature, top_part, info_before, confusion, True)
