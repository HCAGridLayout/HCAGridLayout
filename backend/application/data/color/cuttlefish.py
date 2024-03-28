from colormath.color_objects import sRGBColor, LCHabColor
from colormath.color_conversions import convert_color
import numpy as np

def getHueList(rgbs, mode=1):
    hue_list = []
    for rgb in rgbs:
        rgb_color = sRGBColor(rgb[0] / mode, rgb[1] / mode, rgb[2] / mode)
        lch_color = convert_color(rgb_color, LCHabColor)
        hue_list.append(lch_color.lch_h)
    return hue_list


class CuttleFish:
    def __init__(self, alpham=60, betam=30, color_example=(29, 202, 232)):
        self.alpha_max = alpham
        self.beta_max = betam
        rgb_color = sRGBColor(color_example[0] / 255, color_example[1] / 255, color_example[2] / 255)
        lch_color = convert_color(rgb_color, LCHabColor)
        self.c_default = lch_color.lch_c
        self.l_default = lch_color.lch_l
        print("Init cuttle fish: l = {} and c = {}".format(self.l_default, self.c_default))
    
    def set_args(self, alpham, betam):
        self.alpha_max = alpham
        self.beta_max = betam

    def fit_top(self, item_numbers):
        if np.max(item_numbers) - 1 == 0:
            self.alpha_max = 0.0
            self.beta_max = 360.0 / len(item_numbers)
        else:
            n = (np.max(item_numbers) - 1) * 0.5
            total_part = np.sum(item_numbers) + len(item_numbers) * (n - 1)
            # if total is 360
            self.alpha_max = 360.0 / total_part
            self.beta_max = n * self.alpha_max

        def getRGB(h):
            # print(self.l_default, self.c_default, h)
            hcl_color = LCHabColor(self.l_default, self.c_default, h)
            rgb_color = convert_color(hcl_color, sRGBColor)
            return [rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b]
        
        pcolors = []
        colors = []
        cur_h = 0
        start_h = 0
        for num in item_numbers:
            start_h = cur_h
            for _ in range(num):
                colors.append(getRGB(cur_h))
                cur_h += self.alpha_max
            ph = (start_h + cur_h) / 2
            pcolors.append(getRGB(ph))
            cur_h += self.beta_max
        return pcolors + colors
            

    def fit(self, hue_list, item_numbers, weights=None, dynamic=True):
        assert(len(hue_list) == len(item_numbers))

        if dynamic:
            # all_hue = len(hue_list) * self.beta_max + (np.sum(item_numbers) - len(hue_list)) * self.alpha_max
            # if all_hue > 360:
            #     r = 360 / all_hue
            #     self.alpha_max *= r
            #     self.beta_max *= r

            if np.max(item_numbers) - 1 == 0:
                self.alpha_max = 0.0
                self.beta_max = 360.0 / len(item_numbers)
            else:
                n = (np.max(item_numbers) - 1) * 0.5
                total_part = np.sum(item_numbers) + len(item_numbers) * (n - 1)
                # if total is 360
                self.alpha_max = 360.0 / total_part
                self.beta_max = n * self.alpha_max
            # print("alpha_max: {}, beta_max: {}".format(self.alpha_max, self.beta_max))

        hue_list = hue_list[:]
        for i in range(len(hue_list)):
            hue_list[i] = hue_list[i] % 360
        arg = np.argsort(hue_list)
        
        n = len(hue_list)
        s_beta = 360 / n
        beta = min(s_beta, self.beta_max)
        sum_num = 0
        for number in item_numbers:
            assert(number >= 1)
            sum_num += (number - 1)
        s_alpha = (360 - n * beta) / sum_num
        alpha = min(s_alpha, self.alpha_max)
        
        h_res = []
        sum_m = 0
        sum_gamma = 0
        sum_weight = 0
        if weights is None:
            weights = []
            for i, H in enumerate(hue_list):
                weights.append([1] * item_numbers[i])
        
        start_pos = {}
        sum_id = 0
        for i, pos in enumerate(arg):
            H = hue_list[pos]
            start_pos[pos] = sum_id
            for j in range(item_numbers[pos]):
                hij = i * beta + sum_m * alpha + j * alpha
                sum_gamma += (H-hij) * weights[pos][j]
                sum_weight += weights[pos][j]
                h_res.append(hij)
            sum_m += (item_numbers[pos] - 1)
            sum_id += item_numbers[pos]

        gamma = sum_gamma / sum_weight
        for i in range(len(h_res)):
            h_res[i] += gamma
            if h_res[i] < 0:
                h_res[i] += 360

        # print(h_res)
        res = []
        def getRGB(h):
            # print(self.l_default, self.c_default, h)
            h = h % 360
            hcl_color = LCHabColor(self.l_default, self.c_default, h)
            rgb_color = convert_color(hcl_color, sRGBColor)
            return [rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b]
        
        # from IPython import embed
        # embed()
        
        for hue in hue_list:
            res.append(getRGB(hue))

        for i in range(n):
            start = start_pos[i]
            end = start_pos[i] + item_numbers[i]
            for j in range(start, end):
                res.append(getRGB(h_res[j]))

        return res





