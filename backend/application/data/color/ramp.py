import os, csv, time, random, math
import numpy as np
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor, LCHabColor
from colormath.color_conversions import convert_color

from scipy.interpolate import splprep, splev
from scipy.interpolate import BSpline

# ------------------this part from paper: Color Crafting: Automating the Construction of Designer Quality Color Ramps
# This function breaks up a ramp into its x, y, and z values
def get_xs_ys_zs_from_ramp(ramp):
    xs = []
    ys = []
    zs = []
    for color in ramp:
        xs.append(color[0])
        ys.append(color[1])
        zs.append(color[2])
    return (xs, ys, zs)

# Returns ramp from x, y, and z values
def get_ramp_from_xs_ys_zs(xs, ys, zs):
    ramp = []
    for i in range(len(xs)):
        ramp.append([xs[i], ys[i], zs[i]])
    return ramp

def resample_ramp(ramp, num_points):
    xs, ys, zs = get_xs_ys_zs_from_ramp(ramp)
    tck, u = splprep([xs, ys, zs], s=0)
    interpolated_points = splev(np.linspace(0, 1, num_points), tck)
    interpolated_xs = interpolated_points[0]
    interpolated_ys = interpolated_points[1]
    interpolated_zs = interpolated_points[2]
    return get_ramp_from_xs_ys_zs(interpolated_xs, interpolated_ys, interpolated_zs)

min_lf = 0.5
def resample_ramp_range(ramp, num_points, center_id, hue_range, l_range = -1):
    xs, ys, zs = get_xs_ys_zs_from_ramp(ramp)
    tck, u = splprep([xs, ys, zs], s=0)

    uc = u[center_id]
    point_ct = splev(uc, tck)
    lab_color = LabColor(point_ct[0], point_ct[1], point_ct[2])
    lch_color = convert_color(lab_color, LCHabColor)
    lf = center_id
    ulf = u[lf]
    while lf > 0:
        lf -= 1
        ulf = u[lf]
        point_lf = splev(ulf, tck)
        lab_color2 = LabColor(point_lf[0], point_lf[1], point_lf[2])
        lch_color2 = convert_color(lab_color2, LCHabColor)
        delta = abs(lch_color.lch_h - lch_color2.lch_h)
        delta = min(delta, 360 - delta)
        delta_l = abs(lch_color.lch_l - lch_color2.lch_l)
        if delta > hue_range or (l_range < 0 or delta_l < l_range):
            ulf = (u[lf + 1] + ulf) / 2
            break
    ulf = max(min_lf, ulf)
    rt = center_id
    urt = u[rt]
    while rt < len(xs) - 1:
        rt += 1
        urt = u[rt]
        point_rt = splev(urt, tck)
        lab_color2 = LabColor(point_rt[0], point_rt[1], point_rt[2])
        lch_color2 = convert_color(lab_color2, LCHabColor)
        delta = abs(lch_color.lch_h - lch_color2.lch_h)
        delta = min(delta, 360 - delta)
        if delta > hue_range or (l_range < 0 or delta_l < l_range):
            urt = (u[rt - 1] + urt) / 2
            break
    # print(hue_range, ulf, urt)
    interpolated_points = splev(np.linspace(ulf, urt, num_points), tck)
    interpolated_xs = interpolated_points[0]
    interpolated_ys = interpolated_points[1]
    interpolated_zs = interpolated_points[2]
    return get_ramp_from_xs_ys_zs(interpolated_xs, interpolated_ys, interpolated_zs)

# ----------------------------------------------------------------------------------------------------------------------
def norm(x):

    return min(1, max(0, x))

class RampGenerator:
    ramp_number = 9

    def __init__(self, model_name='both', pre_path=''):
        self.model_name = model_name
        if model_name == 'kmeans':
            self.lab_libs = self._load_model(os.path.join(pre_path, 'model_curves_kmeans.csv'))
        elif model_name == 'bayesian':
            self.lab_libs = self._load_model(os.path.join(pre_path, 'model_curves_bayesian.csv'))
        else:
            lab_libs1 = self._load_model(os.path.join(pre_path, 'model_curves_kmeans.csv'))
            lab_libs2 = self._load_model(os.path.join(pre_path, 'model_curves_bayesian.csv'))
            self.lab_libs = lab_libs1 + lab_libs2
        self.total_time = 0
        self.default_l = 70.0
        self.default_c = 50.0
    
    def _load_model(self, path):
        lab_libs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.reader(f)
                res_list = list(reader)[1:]
            for res in res_list:
                lab_points = []
                L_range = [1000, -1]
                for point in res:
                    points = list(map(float, point[1:-1].split('|')))
                    points[0] += 100.0
                    lab_points.append([points[0], points[1], points[2]])
                    if points[0] < L_range[0]:
                        L_range[0] = points[0]
                    if points[0] > L_range[1]: 
                        L_range[1] = points[0]
                lab_libs.append(lab_points)
        return lab_libs
    
    def getRamp(self, color, template, hue_range, num, lumi_range=-1):
        # color: colormath lab color
        ramp = self.lab_libs[template]
        color_l = color.lab_l

        # find closest l in ramp
        closest_l = 1000
        closest_index = 0
        for i, point in enumerate(ramp):
            if abs(point[0] - color_l) < abs(closest_l - color_l):
                closest_l = point[0]
                closest_index = i
        delta = color_l - closest_l
        delta_a = color.lab_a - ramp[closest_index][1]
        delta_b = color.lab_b - ramp[closest_index][2]
        
        # get the ramp
        target_ramp = []
        for i, point in enumerate(ramp):
            target_ramp.append([point[0] + delta, point[1] + delta_a, point[2] + delta_b])

        modify_ramp = resample_ramp_range(target_ramp, num, closest_index, hue_range, lumi_range)
        ramp_colors = []
        for point in modify_ramp:
            color = LabColor(point[0], point[1], point[2])
            color = convert_color(color, sRGBColor)
            ramp_colors.append([norm(color.rgb_r), norm(color.rgb_g), norm(color.rgb_b)])
        return ramp_colors
    
    def get_init_colors(self, num):
        delta_hue = 360 / num
        start_hue = random.randint(0, 360)
        colors = []
        for i in range(num):
            color = LCHabColor(self.default_l, self.default_c, start_hue)
            start_hue = (start_hue + delta_hue) % 360
            color = convert_color(color, sRGBColor)
            colors.append([norm(color.rgb_r), norm(color.rgb_g), norm(color.rgb_b)])
        return colors

    def extend_color(self, colors, item_nums):
        self.total_time = 0
        start_time = time.time()
        if colors[0][0] < 0:
            colors = self.get_init_colors(len(item_nums))
        
        hue_weights = []
        for num in item_nums:
            hue_weights.append(math.sqrt(num))

        color_hues = []
        color_lums = []
        for i in range(len(colors)):
            color = sRGBColor(colors[i][0], colors[i][1], colors[i][2])
            color = convert_color(color, LCHabColor)
            color_hues.append(color.lch_h)
            color_lums.append(color.lch_l)

        res_colors = colors[:]

        for i in range(len(colors)):
            min_range = 360
            min_lumin = 100
            for j in range(len(color_hues)):
                if i == j:
                    continue
                range1 = abs(color_hues[i] - color_hues[j])
                range2 = 360 - range1
                gap = max(hue_weights[i], hue_weights[j])
                ratio = hue_weights[i] / (hue_weights[i] + hue_weights[j] + gap)
                min_range = min(min_range, min(range1, range2) * ratio)
                ratio = hue_weights[i] / (hue_weights[i] + hue_weights[j])
                min_lumin = min(min_lumin, abs(color_lums[i] - color_lums[j]) * ratio)
            color = sRGBColor(colors[i][0], colors[i][1], colors[i][2])
            color = convert_color(color, LabColor)
            template_id = random.randint(0, len(self.lab_libs) - 1)
            ramp_colors = self.getRamp(color, template_id, min_range, item_nums[i], min_lumin)
            res_colors += ramp_colors
            # print(len(ramp_colors), item_nums[i])

        end_time = time.time()
        self.total_time += end_time - start_time
        return res_colors
