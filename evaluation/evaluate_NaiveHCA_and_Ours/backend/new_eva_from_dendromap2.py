import numpy as np
import random



import pickle
def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

for dataset in ["cifar100_for_dendromap", "imagenet1k", "inat2021"]:
    for size in ["30px", "20px", "16px"]:
        select = ''
        for ctype in [""]:
            for method in ["qap"]:
                method_name = "Ours"
                if method == "dendromap":
                    method_name = "DendroMap"
                print(dataset, size, method_name)
                ans = load_pickle(method + '_dendroans_' + dataset + select + "_HV" + '_' + size + '.pkl')
                for key in ["zoom"]:
                    mean = {}
                    # print(len(ans[key]))
                    len2 = 0
                    for item in ans[key]:
                        if 'stab-position' in item and item['stab-position'] > 0:
                            len2 += 1
                        for key2 in item:
                            if key2 == "layout":
                                continue
                            if key2 not in mean:
                                mean[key2] = 0
                            if (key2 == 'stab-position' or key2 == 'stab-shape') and item['stab-position'] == 0:
                                continue
                            mean[key2] += item[key2]
                    for key2 in mean:
                        if key2 == 'stab-position' or key2 == 'stab-shape':
                            mean[key2] /= len2+1e-12
                            continue
                        mean[key2] /= len(ans[key])
                    print(key, mean)
