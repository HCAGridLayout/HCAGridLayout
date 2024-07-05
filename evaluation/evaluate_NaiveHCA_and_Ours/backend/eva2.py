import numpy as np
import random



import pickle
def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

avg = {"": {"qap": {}, "tsne": {}}, "_HV": {"qap": {}, "tsne": {}}}
avg_cnt = 0

for dataset in ["cifar100", "imagenet1k", "inat2021"]:
    for size in ["900", "1600", "2500"]:
        select = ''
        avg_cnt += 1
        for ctype in ["", "_HV"]:
            for method in ["qap", "tsne"]:
                method_name = "Ours"
                if method == "tsne":
                    method_name = "NaiveHCA"
                convex_name = "TripleRatio"
                if ctype == "_HV":
                    convex_name = "PerimeterRatio"
                print(dataset, size, method_name, convex_name)
                ans = load_pickle(size + '/' + method + '_ans_' + dataset + '_' + size + select + ctype + '.pkl')
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

                    for key2 in mean:
                        if key2 not in avg[ctype][method]:
                            avg[ctype][method][key2] = 0+mean[key2]
                        else:
                            avg[ctype][method][key2] += mean[key2]
print()
print("------------avg-------------")
for ctype in ["", "_HV"]:
    for method in ["qap", "tsne"]:
        method_name = "Ours"
        if method == "tsne":
            method_name = "NaiveHCA"
        convex_name = "TripleRatio"
        if ctype == "_HV":
            convex_name = "PerimeterRatio"
        print("Compare with NaiveHCA:", method_name, convex_name)
        mean = avg[ctype][method].copy()
        for key2 in mean:
            mean[key2] /= avg_cnt
        print(mean)

