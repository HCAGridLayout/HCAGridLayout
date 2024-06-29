from application.data.port import Port
import numpy as np
import random
import os
import time

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


for dataset in ["cifar100", "imagenet1k", "inat2021"]:
    for size in [900, 1600, 2500]:
        select_ratio = 0.35
        for select_method in ["square"]:
            for use_HV in [False, True]:
                for method in ["qap", "tsne"]:
                    ctype = ""
                    if use_HV:
                        ctype = "_HV"
                    select_str = ""
                    if select_method == "cluster":
                        select_str = "_cluster"
                    if os.path.exists(str(size) + '/' + method + '_ans_' + dataset + '_' + str(size) + select_str + ctype + '.pkl'):
                        continue

                    port = Port(size, {"use_HV": use_HV, "use_conf": True, "scenario": "ans", "select_method": select_method, "method": method})
                    port.load_dataset(dataset) #'MNIST''cifar'
                    select_type = select_method

                    mt0 = 5
                    mt2 = 5
                    if (size>400)and(dataset=="cifar100"):
                        mt0 = 30
                        mt2 = 0

                    for t0 in range(30):
                        folder_path = "cache/"+dataset
                        file_list = os.listdir(folder_path)
                        for file_name in file_list:
                            file_path = os.path.join(folder_path, file_name)
                            os.remove(file_path)

                        gridlayout = port.top_gridlayout()

                        if t0 >= mt0:
                            continue

                        for t1 in range(5):

                            samples = select(gridlayout, select_ratio, select_type)

                            gridlayout2 = port.layer_gridlayout(gridlayout['index'], samples)

                            for t2 in range(mt2):

                                samples = select(gridlayout2, select_ratio, select_type)

                                gridlayout3 = port.layer_gridlayout(gridlayout2['index'], samples)
                                _ = port.zoom_out_gridlayout(gridlayout3['index'])

                            _ = port.zoom_out_gridlayout(gridlayout2['index'])

