from application.data.port import Port
import numpy as np
import random
import os
import json
from scipy.spatial.distance import cdist
import time
from application.utils.pickle import *
import math

# 修改LabelHierarchy.py 分级读取

def select(gridlayout, select_ratio=0.35, type='square'):
    random.seed(int(time.time()))
    if type == 'full':
        return np.arange(len(gridlayout['labels']), dtype='int')

    square_len = round(np.sqrt(len(gridlayout['grid'])))

    if type == 'square':
        size = round(select_ratio * square_len)
        x1 = random.randint(0, square_len - size)
        x2 = x1 + size
        y1 = random.randint(0, square_len - size)
        y2 = y1 + size
        print("selected x y", x1, y1)

        samples = []
        part_list = []
        for i in range(x1, x2):
            for j in range(y1, y2):
                if gridlayout['grid'][i * square_len + j] < len(gridlayout['labels']):
                    id = gridlayout['grid'][i * square_len + j]
                    samples.append(id)
                    if gridlayout['part_labels'][id] not in part_list:
                        part_list.append(gridlayout['part_labels'][id])

        samples = np.array(samples)
        return samples

# dataset_name = "cifar100"
# dataset_name = "imagenet1k"
# dataset_name = "inat2021"
for dataset_name in ["cifar100", 'imagenet1k', 'inat2021']:
# for dataset_name in ["cifar100"]:
    dataset = dataset_name
    if dataset_name == 'cifar100':
        dataset = 'cifar100_for_dendromap'
    for image_size in [30, 20, 16]:
    # for image_size in [20]:

        if os.path.exists('qap_dendroans_' + dataset + "_HV" + '_' + str(image_size) + 'px.pkl'):
            continue

        with open("dendromap_step_"+dataset_name+"_"+str(image_size)+"px.json", 'r') as f:
            dendromap_step = json.load(f)

        print(len(dendromap_step))

        port = None
        gridlayout_stack = []
        dendromap_stack = []

        for i in range(len(dendromap_step)):
            dendro = dendromap_step[i]
            if dendro["method"] == "top":
                folder_path = "cache/" + dataset
                if os.path.exists(folder_path):
                    file_list = os.listdir(folder_path)
                    for file_name in file_list:
                        file_path = os.path.join(folder_path, file_name)
                        os.remove(file_path)

                gridlayout_stack = []
                dendromap_stack = []

                num = math.ceil(np.sqrt(len(dendro["samples"]))) ** 2

                np.save("next_num", [num])
                port = Port(num, {"use_HV": True, "use_conf": True, "scenario": "dendroans", "select_method": "square", "method": "qap", "px": image_size})
                port.load_dataset(dataset)

                new_gridlayout = port.top_gridlayout()
                gridlayout_stack.append(new_gridlayout)

                np.save("next_num", [-1])

                ids = []
                for item in dendro["samples"]:
                    id = int(item["id"])
                    ids.append(id)

                dendromap_stack.append({'samples': dendro["samples"], 'ids': ids})

            elif dendro["method"] == "zoomin":
                gridlayout_bf = gridlayout_stack[-1]

                # square_len1 = round(np.sqrt(len(gridlayout_bf['grid'])))
                # show_grid(gridlayout_bf['grid'], gridlayout_bf['part_labels'], square_len1, 'selected1.png', just_save=True)

                selected = []
                ids = []
                for item in dendro["samples"]:
                    id = int(item["id"])
                    ids.append(id)
                    if id in dendromap_stack[-1]['ids']:
                        selected.append(id)

                num = round(np.sqrt(len(dendro["samples"]))) ** 2
                np.save("next_num", [num])

                for tt in range(5):
                    random.seed(int(time.time()))
                    print("selected", len(selected), len(dendromap_stack[-1]['ids']))
                    samples = select(gridlayout_bf, np.sqrt(len(selected)/len(dendromap_stack[-1]['ids'])), 'square')

                    # if_selected = np.zeros(len(gridlayout_bf['labels']), dtype='int')
                    # if_selected[samples] = 1
                    # show_grid(gridlayout_bf['grid'], if_selected, square_len1, 'selected2.png', just_save=True)

                    new_gridlayout = port.layer_gridlayout(gridlayout_bf['index'], samples)

                    if tt < 4:
                        _ = port.zoom_out_gridlayout(new_gridlayout['index'])
                    else:
                        gridlayout_stack.append(new_gridlayout)

                        ids = []
                        for item in dendro["samples"]:
                            id = int(item["id"])
                            ids.append(id)

                        dendromap_stack.append({'samples': dendro["samples"], 'ids': ids})

                np.save("next_num", [-1])
            else:
                gridlayout_bf = gridlayout_stack[-1]
                _ = port.zoom_out_gridlayout(gridlayout_bf['index'])
                gridlayout_stack.pop()
                dendromap_stack.pop()
