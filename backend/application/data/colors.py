import numpy as np
from .LabelHierarchy import LabelHierarchy
from .color.ramp import RampGenerator
from .color.palettailor import palettailor
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
from application.data.color.colorlib import find_palette_global, find_palette_global_grid
import time
from copy import deepcopy
from .color.cuttlefish import CuttleFish, getHueList
from .grid_plot import show_grid
import os

def norm(x):
    return min(1, max(0, x))

save_time = []

class MyColorMap(object):
    modes = ['colorlib', 'ramp', 'tree', 'palettailor', 'cuttlefish']
    def __init__(self) -> None:
        self.extend_mode = self.modes[0]
        self.seed = 0
        self.final_extend = False
        self.colorlib_type = 'd' # 'd' 's' d:discriminative s:similar
        self.similar_arg = 0.2
        self.rgbs = [
            (248, 182, 187),
            (171, 227, 246),
            (216, 193, 246),
            (243, 228, 191),
            (185, 243, 190),
            (252, 245, 155),
            (221, 221, 221),
            (138, 170, 208),
            (191, 185, 134),
            (255, 193, 152),
            (127, 137, 253),
            (255, 136, 104),
            (175, 203, 191),
            (170, 167, 188),
            (254, 228, 179)
        ]
        self.rgbs1 = list(map(lambda x: list(map(lambda v: v / 255, x)), self.rgbs))
        self.ramp = RampGenerator('kmeans', './application/data/color/')
        self.cuttle = CuttleFish()

    def getTime(self, num):
        assert(num <= len(save_time))
        return np.mean(save_time[-num:]), np.std(save_time[-num:])
    
    def colorSet(self, sets):
        ret = []
        for i in sets:
            ret.append(self.rgbs1[i])
        return ret
    
    def hierarchyColorSet(self, top_labels, labels, hierarchy: LabelHierarchy, history=None):
        # without history, assign colors by top_labels from self.rgbs1
        if history is None:
            records = {}
            records['undivided'] = []
            cur_color_id = 0
            insides_colors = {}
            lv1_labels = np.unique(top_labels).tolist()
            for i, label in enumerate(lv1_labels):
                record = {
                    'label': label,
                    'color_id': '{}'.format(cur_color_id),
                    'divided': False,
                }
                cur_color_id += 1
                records[label] = record
                records['undivided'].append(label)
                insides_colors[label] = []
            for i, label in enumerate(labels):
                tlabel = top_labels[i]
                if top_labels[i] == label:
                    continue
                if label not in insides_colors[tlabel]:
                    insides_colors[tlabel].append(label)
            for label in insides_colors:
                self.interplotColors(records, label, insides_colors[label])
            records['cur_color_id'] = cur_color_id
            return records
        
        # with history, assign colors by hierarchy
        records = history
        cur_color_id = records['cur_color_id']
        cur_labels = np.unique(labels).tolist()
        insides_colors = {}
        for i, label in enumerate(cur_labels):
            path, dist = hierarchy.get_path_to_root(label)
            find = False
            for j, l in enumerate(path):
                if l in records:
                    if j > 0:
                        if l not in insides_colors:
                            insides_colors[l] = []
                        insides_colors[l].append((label, j))
                    find = True
                    break
            if not find:
                record = {
                    'label': label,
                    'divided': False,
                    'color_id': '{}'.format(cur_color_id)
                }
                cur_color_id += 1
                records[label] = record
                records['undivided'].append(label)


        for label in insides_colors:
            insides_colors[label].sort(key=lambda x: x[1])
            cur_inside = []
            if 'insides' in records[label]:
                for i in records[label]['insides']:
                    if i in cur_labels:
                        cur_inside.append(i)
            for i in insides_colors[label]:
                if i[0] not in cur_inside:
                    cur_inside.append(i[0])
            self.interplotColors(records, label, cur_inside)

        records['cur_color_id'] = cur_color_id
        return records 

    def interplotColors(self, records, label, inside_labels):
        if len(inside_labels) == 0:
            return
        record = records[label]
        label_color_id = record['color_id']
        child_id = 0
        for ilabel in inside_labels:
            irecord = {
                'label': ilabel,
                'divided': False,
                'color_id': '{}-{}'.format(label_color_id, child_id)
            }
            child_id += 1
            records[ilabel] = irecord
            records['undivided'].append(ilabel)

        record['divided'] = True
        if label in records['undivided']:
            records['undivided'].remove(label)
        if 'insides' in record:
            for ilabel in record['insides']:
                if ilabel not in inside_labels:
                    del records[ilabel]
        record['insides'] = inside_labels
        
    def hierarchyColorSet2(self, top_labels, labels, partition_labels, history=None, label_hierarchy=None, grid_info=None):
        # use top_label as color set standard
        # self.ramp.total_time = 0
        records = {}
        cur_color_id = 0
        if history is None:
            cur_color_id = 0
        else:
            cur_color_id = history['cur_color_id']
        records['single'] = []
        
        insides_colors = {}
        lv1_labels = np.unique(top_labels).tolist()
        lv2_labels = np.unique(labels).tolist()
        no_hierarchy = False
        min_index = min(lv1_labels)
        if len(lv1_labels) == len(lv2_labels) and history is None:
            no_hierarchy = True
        tree = label_hierarchy
        
        for i, label in enumerate(lv1_labels):
            insides_colors[label] = [[], []]
            if history is not None and label in history:
                records[label] = deepcopy(history[label])
                continue
            record = {
                'label': label,
                'color_id': '{}'.format(cur_color_id),
                'color': [-1,-1,-1] if self.extend_mode != 'tree' else tree['hierarchy'][tree['id2label'][label]]['rgb']
            }
            record['parent_color'] = record['color']
            cur_color_id += 1
            records[label] = record

        for i, label in enumerate(labels):
            tlabel = top_labels[i]
            if no_hierarchy:
                tlabel = min_index
            elif top_labels[i] == label:
                if label not in records['single']:
                    records['single'].append(label)
            if label not in insides_colors[tlabel][0]:
                insides_colors[tlabel][0].append(label)
                insides_colors[tlabel][1].append((label, partition_labels[i]))
        records['cur_color_id'] = cur_color_id

        if self.extend_mode == 'tree':
            for label in insides_colors:
                self.treeColors(records, label, insides_colors[label][1], label_hierarchy)
        else:
            self.extend = {
                'colors': [],
                'numbers': [],
                'plabels': [],
                'labels': []
            }
            for label in insides_colors:
                self.interplotColors_hyper(records, label, insides_colors[label][1])

        if self.extend_mode == 'tree' or len(self.extend['colors']) == 0:
            return records
        
        if self.extend_mode == 'ramp':
            items_num = []
            for elabels in self.extend['labels']:
                items_num.append(len(elabels))
            palette = self.ramp.extend_color(self.extend['colors'], items_num)
            print(len(palette))
            idx0 = 0
            idx1 = 0
            for label in self.extend['plabels']:
                records[label]['color'] = palette[idx0]
                idx0 += 1
            for elabels in self.extend['labels']:
                for label in elabels:
                    records[label]['color'] = palette[idx0]
                    records[label]['parent_color'] = palette[idx1]
                    idx0 += 1
                idx1 += 1
            save_time.append(self.ramp.total_time)

        if self.extend_mode == 'colorlib':
            # prepare data driven
            driven_data = None
            if grid_info is not None:
                driven_data = {}
                driven_data['size'] = grid_info['size']
                driven_data['grid'] = grid_info['grid']
                driven_data['labels'] = grid_info['labels']
                driven_data['quality'] = 0.99 # 0.98 if self.colorlib_type == 'd' else 0.99
                driven_data['mode'] = 0 if self.colorlib_type == 'd' else 1   
                label_list = []
                for elabels in self.extend['labels']:
                    for label in elabels:
                        label_list.append(label)
                normlist = np.sort(np.unique(driven_data['labels']))
                curlist = np.array(label_list)
                curpermutation = np.zeros(len(curlist), dtype=int)
                for i in range(len(curlist)):
                    curpermutation[i] = np.where(curlist[i] == normlist)[0][0]
                # curpermutation = curpermutation.astype(int)
                # from IPython import embed; embed()
                try:
                    assert(np.all(curlist == normlist[curpermutation]))
                except:
                    from IPython import embed; embed();
                driven_data['label_list'] = label_list
                # from IPython import embed; embed();
                driven_data['similarity'] = grid_info['similarity'][curpermutation][:, curpermutation]


                time_saves = []
                for _ in range(1):
                    start_time = time.time()
                    start_colors = self.extend['colors']
                    if start_colors[0][0] < 0:
                        start_colors = []
                    # inputs = [start_colors, self.extend['numbers'], driven_data['size'], driven_data['size'], driven_data['grid'], driven_data['labels'], driven_data['label_list'], driven_data['similarity']]
                    # # save pickle
                    # import pickle
                    # with open('color_input.pkl', 'wb') as f:
                    #     pickle.dump(inputs, f)
                    # from IPython import embed; embed()
                    palettes = find_palette_global_grid(start_colors, self.extend['numbers'], 
                                                            driven_data['size'], driven_data['size'], driven_data['grid'], driven_data['labels'], 
                                                            driven_data['label_list'], driven_data['similarity'], [0,360], 0, 48, 0.99, 12, 2)
                    # print(np.round(grid_info['similarity'],3))
                    end_time = time.time()
                    time_saves.append(end_time - start_time)
                    print("find_palette_global", end_time - start_time)
                    # rt_scores.append(palettes[-1])


                # rt_scores = np.array(rt_scores).T
                # save_record = []
                # for i in range(rt_scores.shape[0]):
                #     save_record.append(np.mean(rt_scores[i]))
                #     save_record.append(np.std(rt_scores[i]))
                #     save_record.append(np.min(rt_scores[i]))
                # save_records.append(save_record)
                # print(save_records)

                # import csv
                # with open('save_records.csv', 'w', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     for row in save_records:
                #         writer.writerow(row)
                # total_end_time = time.time()
                # print("total_time", total_end_time - total_start_time)
            
            # save palettes as tmp.pkl
            # import pickle
            # with open('tmp.pkl', 'wb') as f:
            #     pickle.dump(palettes, f)
                
            # save_start = 0
            # stage = 1
            # while True:
            #     path = '__grid/{}.png'.format(save_start)
            #     if not os.path.exists(path):
            #         break
            #     save_start += 1
            # idx0 = 0
            # for label in self.extend['plabels']:
            #     idx0 += 1
            # start_idx0 = idx0
            # for i in range(stage):
            #     color_map = {}
            #     for elabels in self.extend['labels']:
            #         for label in elabels:
            #             color_map[label] = palettes[idx0]
            #             idx0 += 1
            #     # show_grid(driven_data['grid'], driven_data['labels'], driven_data['size'], color_map, path='__grid/{}.png'.format(save_start + i), just_save=True)
            #     print(i, idx0,len(palettes))
            # cur_value = (idx0 - start_idx0) // stage
            # # print(idx0, start_idx0,"cur_value", cur_value)
            # start_idx0 += (stage - 1) * cur_value

            
            
                # rt_scores = np.array(rt_scores).T
                # save_record = []
                # for i in range(rt_scores.shape[0]):
                #     save_record.append(np.mean(rt_scores[i]))
                #     save_record.append(np.std(rt_scores[i]))
                #     save_record.append(np.min(rt_scores[i]))
                # save_records.append(save_record)
                # print(save_records)

                # import csv
                # with open('save_records.csv', 'w', newline='') as csvfile:
                #     writer = csv.writer(csvfile)
                #     for row in save_records:
                #         writer.writerow(row)
                # total_end_time = time.time()
                # print("total_time", total_end_time - total_start_time)
            
            # save palettes as tmp.pkl
            # import pickle
            # with open('tmp.pkl', 'wb') as f:
            #     pickle.dump(palettes, f)
                
            # save_start = 0
            # stage = 1
            # while True:
            #     path = '__grid/{}.png'.format(save_start)
            #     if not os.path.exists(path):
            #         break
            #     save_start += 1
            # idx0 = 0
            # for label in self.extend['plabels']:
            #     idx0 += 1
            # start_idx0 = idx0
            # for i in range(stage):
            #     color_map = {}
            #     for elabels in self.extend['labels']:
            #         for label in elabels:
            #             color_map[label] = palettes[idx0]
            #             idx0 += 1
            #     # show_grid(driven_data['grid'], driven_data['labels'], driven_data['size'], color_map, path='__grid/{}.png'.format(save_start + i), just_save=True)
            #     print(i, idx0,len(palettes))
            # cur_value = (idx0 - start_idx0) // stage
            # # print(idx0, start_idx0,"cur_value", cur_value)
            # start_idx0 += (stage - 1) * cur_value

            
            # print("find_palette_global", end_time - start_time)
            save_time.append(end_time - start_time)
            # if len(save_time) >= 10:
            #     print("find_palette_global", np.mean(save_time[-10:]), np.std(save_time[-10:]))
            idx0 = 0
            idx1 = 0
            for label in self.extend['plabels']:
                records[label]['color'] = palettes[idx0]
                idx0 += 1
            # idx0 = start_idx0
            for elabels in self.extend['labels']:
                for label in elabels:
                    records[label]['color'] = palettes[idx0]
                    records[label]['parent_color'] = palettes[idx1]
                    idx0 += 1
                idx1 += 1

        if self.extend_mode == 'palettailor':
            inside_labels = []
            for elabels in self.extend['labels']:
                inside_labels.append(np.array(elabels).tolist())
            result = palettailor(self.extend['colors'], inside_labels, [grid_info['size'], grid_info['size']],
                                 grid_info['grid'].tolist(), grid_info['labels'].tolist(), self.seed)
            palettes = result['colors']
            run_time = result['time']
            print("palettailor", run_time)
            idx0 = 0
            idx1 = 0
            for label in self.extend['plabels']:
                records[label]['color'] = palettes[idx0]
                idx0 += 1
            for elabels in self.extend['labels']:
                for label in elabels:
                    records[label]['color'] = palettes[idx0]
                    records[label]['parent_color'] = palettes[idx1]
                    idx0 += 1
                idx1 += 1
            save_time.append(run_time)

        if self.extend_mode == 'cuttlefish':
            palette = None
            run_time = 0
            items_num = []
            for elabels in self.extend['labels']:
                items_num.append(len(elabels))

            if self.extend['colors'][0][0] < 0:
                start_time = time.time()
                palette = self.cuttle.fit_top(items_num)
                run_time = time.time() - start_time
            else:
                hue_list = getHueList(self.extend['colors'])
                start_time = time.time()
                palette = self.cuttle.fit(hue_list, items_num)
                run_time = time.time() - start_time

            print("CuttleFish", run_time)
            idx0 = 0
            idx1 = 0
            for label in self.extend['plabels']:
                records[label]['color'] = palette[idx0]
                idx0 += 1
            for elabels in self.extend['labels']:
                for label in elabels:
                    records[label]['color'] = palette[idx0]
                    records[label]['parent_color'] = palette[idx1]
                    idx0 += 1
                idx1 += 1
            save_time.append(run_time)
        
        # from IPython import embed; embed();
        # print("total_time", self.ramp.total_time)
        return records

    def interplotColors_hyper(self, records, label, inside_labels):
        if len(inside_labels) == 0:
            return
        
        record = records[label]
        label_color_id = record['color_id']
        ilabels = []
        for child_id, ilabel in enumerate(inside_labels):
            ilabel = ilabel[0]
            ilabels.append(ilabel)
            irecord = {
                'label': ilabel,
                'color_id': '{}-{}'.format(label_color_id, child_id),
            }
            records[ilabel] = irecord

        self.extend['colors'].append(record['color'])
        self.extend['numbers'].append(len(inside_labels))
        self.extend['plabels'].append(label)
        self.extend['labels'].append(ilabels)

    
    def treeColors(self, records, label, inside_labels, tree):
        if len(inside_labels) == 0:
            return
    
        record = records[label]
        label_color_id = record['color_id']
        ilabels = []
        for child_id, ilabel in enumerate(inside_labels):
            ilabel = ilabel[0]
            ilabels.append(ilabel)
            irecord = {
                'label': ilabel,
                'color_id': '{}-{}'.format(label_color_id, child_id),
                'color': tree['hierarchy'][tree['id2label'][ilabel]]['rgb'],
                'parent_color': record['color']
            }
            records[ilabel] = irecord
