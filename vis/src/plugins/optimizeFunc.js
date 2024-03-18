/* eslint-disable */
import {matsuda_templates, geometric_hue_templates} from './hue_templates.js';
import geo_lc_linear from './geometric_lc.js';
import { color } from 'd3';
export {
    evaluatePalette,
    simulatedAnnealing2FindBestPalette,
};
import ColorScope from './color_scope.js';
var glo_scope = new ColorScope();
var harmony_judger_matsuda = new matsuda_templates();
var harmony_judger_geo_hue = new geometric_hue_templates(); 
var harmony_mode = 'matsuda'; // 'matsuda';
var harmony_judger = null;
if (harmony_mode === 'matsuda') {
    harmony_judger = harmony_judger_matsuda;
} else if (harmony_mode === 'geometric') {
    harmony_judger = harmony_judger_geo_hue;
    // harmony_judger = harmony_judger_matsuda;
}

var base_rgb = null;
var best_color_obj = null;
// var best_color_dis = 15;
var best_color_dis = 60;
var min_best_color_dis = 8;
var exclude_margin = 8;

var best_count = 0;
var random_extend = false;
var global_color_dis = 20; // min dis between two color, default: 10 // 20

c3.load('./static/lib/c3_data.json');
// color name lookup table
var color_name_map = {};
for (var c = 0; c < c3.color.length; ++c) {
    var x = c3.color[c];
    color_name_map[[x.L, x.a, x.b].join(",")] = c;
}
// console.log('c3', c3.color)
window.color_name_map = color_name_map;

var bgcolor = "#fff";
var score_importance_weight = [1, // data driven: color discrimination / class similarity
    1,  // color name difference
    1,  // color discrimination constraint
    2];  // color harmony
var shift_p = 0.5;

var global_count = 0; // if there are more than <global_count> colors that are not in selected color names, then discard this palette  default: 2
var global_dec = 0.99; // 0.99: high efficiency   0.999: high qualit
var max_count = 100;
var seed = 45; //45
function random() {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}

/**
 * functions for data driven
 */
var data_driven_info = {};
var use_data_driven_sa = true;
var use_post = false;
var use_cross_opt = true;
var cross_args = [1, 1]; // times and num [2, 2]
var data_driven_score = null;
var neighbor_num = 8; // 4 or 8
var similarity_type = 0; // 0: in class, 1: in neighbor class
function initDataDriven(data, labels) { 
    // console.log(data, labels);
    // console.log("init data driven");
    data_driven_info = {};  
    if (data == null || labels == null || score_importance_weight[0] === 0) {
        data_driven_info['use'] = false;
        shift_p = 1;
        data_driven_score = function() { return 0; }
        return;
    }
    data_driven_info['use'] = true;
    data_driven_info['type'] = data['type'];
    data_driven_info['labels'] = labels;
    let label_set = new Set(labels);
    data_driven_info['set'] = label_set;
    data_driven_info['data'] = data['grids'].filter(grid => label_set.has(grid.label));
    let label_map = {};
    labels.forEach((label, idx) => {
        label_map[label] = idx;
    });
    data_driven_info['label_map'] = label_map;
    shift_p = 0.5;

    if (data['type'] === 'discri') {
        data_driven_score = data_driven_discri;
        data_driven_info['args'] = init_discri(data_driven_info['data'] , labels, label_map, label_set);
    } else {
        data_driven_info['label_map2dist'] = data['meta']['label_map'];
        data_driven_info['label_dists'] = data['meta']['label_dists'];
        data_driven_info['label_centers'] = labels.map(label => {
            let label_grids = data['grids'].filter(grid => grid.label === label);
            let label_sum = label_grids.reduce((sum, grid) => {
                sum[0] += grid.x;
                sum[1] += grid.y;
                return sum;
            }, [0, 0]);
            return [label_sum[0] / label_grids.length + label_grids[0].width / 2, 
                label_sum[1] / label_grids.length + label_grids[0].height / 2];
        });
        data_driven_info['center'] = [data_driven_info['label_centers'].reduce((sum, center) => sum + center[0], 0) / data_driven_info['label_centers'].length,
            data_driven_info['label_centers'].reduce((sum, center) => sum + center[1], 0) / data_driven_info['label_centers'].length];
        
        if (similarity_type === 0) {
            data_driven_score = data_driven_similarity;
        } else {
            data_driven_info['args'] = init_neigh_similar(data_driven_info['data'], labels, data_driven_info['label_map'], 
                data_driven_info['label_map2dist'], label_set, data_driven_info['label_dists']);
            data_driven_score = data_driven_similarity2;
        }
    } 
}

function init_discri(grids, labels, label_map, label_set) {
    let args = labels.map(() => labels.map(() => 0));
    grids.forEach((grid, idx) => {
        let n = 0;
        let label_id = label_map[grid.label];
        let res_labels = [];
        let res_dist = [];
        let judgelabel = function (label, dist) {
            if (label_set.has(label) && label_map[label] !== label_id) {
                res_labels.push(label_map[label]);
                res_dist.push(dist);
                n += 1;
            }
        };
        if (grid.top !== undefined) {
            judgelabel(grid.top.label, 1);
            if (neighbor_num === 8) {
                if (grid.top.left !== undefined) {
                    judgelabel(grid.top.left.label, Math.SQRT1_2);
                }
                if (grid.top.right !== undefined) {
                    judgelabel(grid.top.right.label, Math.SQRT1_2);
                }
            }
        }
        if (grid.bottom !== undefined) {
            judgelabel(grid.bottom.label, 1);
            if (neighbor_num === 8) {
                if (grid.bottom.left !== undefined) {
                    judgelabel(grid.bottom.left.label, Math.SQRT1_2);
                }
                if (grid.bottom.right !== undefined) {
                    judgelabel(grid.bottom.right.label, Math.SQRT1_2);
                }
            }
        }
        if (grid.left !== undefined) {
            judgelabel(grid.left.label, 1);
        }
        if (grid.right !== undefined) {
            judgelabel(grid.right.label, 1);
        }
        if (n > 0) {
            let delta = 1 / n;
            for (let i = 0; i < res_labels.length; i++) {
                args[res_labels[i]][label_id] += delta * res_dist[i];
                args[label_id][res_labels[i]] += delta * res_dist[i];
            }
        }
    });
    args = args.map(row => row.map(val => val / grids.length));
    return args;
}

function data_driven_discri(palette) {
    let args = data_driven_info['args'];
    let dis = 0;
    // from args calculate
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            dis += d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j])) * args[i][j];
        }
    }
    return dis;
};

var epsilon = 1e-4;
function g(similarity) {
    // change similarity from (-1,1) to (-inf, 0)
    return Math.log(1 - 0.5 * (1 + similarity));
}

function data_driven_similarity(palette, func = g) {
    let labels = data_driven_info['labels'];
    let label_map = data_driven_info['label_map2dist'];
    let matrix = data_driven_info['label_dists'];
    let dis = 0;
    for (let i = 0;i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            let color_dist = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j])); // Math.abs(palette[i].h - palette[j].h); 
            let label_dist = matrix[label_map[labels[i]]][label_map[labels[j]]];
            // dis += (1 - label_dist) / color_dist;
            dis += func(1 - label_dist) * color_dist;
        }
    }
    dis /= (palette.length * (palette.length - 1) * 0.5);
    return dis;
}

function init_neigh_similar(grids, labels, label_map, label_map2dist, label_set, dist_matrix, func = g) {
    let args = labels.map(() => labels.map(() => 0));
    grids.forEach((grid, idx) => {
        let n = 0;
        let label_id = label_map[grid.label];
        let label_dist_id = label_map2dist[grid.label];
        let res_labels = [];
        let res_dist = [];
        let judgelabel = function (label) {
            if (label_set.has(label) && label_map[label] !== label_id) {
                res_labels.push(label_map[label]);
                res_dist.push(func(1 - dist_matrix[label_map2dist[label]][label_dist_id]));
                n += 1;
            }
        };
        if (grid.top !== undefined) {
            judgelabel(grid.top.label);
            if (neighbor_num === 8) {
                if (grid.top.left !== undefined) {
                    judgelabel(grid.top.left.label);
                }
                if (grid.top.right !== undefined) {
                    judgelabel(grid.top.right.label);
                }
            }
        }
        if (grid.bottom !== undefined) {
            judgelabel(grid.bottom.label);
            if (neighbor_num === 8) {
                if (grid.bottom.left !== undefined) {
                    judgelabel(grid.bottom.left.label);
                }
                if (grid.bottom.right !== undefined) {
                    judgelabel(grid.bottom.right.label);
                }
            }
        }
        if (grid.left !== undefined) {
            judgelabel(grid.left.label);
        }
        if (grid.right !== undefined) {
            judgelabel(grid.right.label);
        }
        if (n > 0) {
            let delta = 1 / n;
            for (let i = 0; i < res_labels.length; i++) {
                args[res_labels[i]][label_id] += delta * res_dist[i];
                args[label_id][res_labels[i]] += delta * res_dist[i];
            }
        }
    });
    args = args.map(row => row.map(val => val / grids.length));
    return args;
}

function data_driven_similarity2(palette) {
    let args = data_driven_info['args'];
    let labels = data_driven_info['labels'];
    let label_map = data_driven_info['label_map'];
    let dis = 0;
    // from args calculate
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            dis += d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j])) * args[label_map[labels[i]]][label_map[labels[j]]];
        }
    }
    return dis;
}


/**
 * calculating the Color Saliency 
 * reference to "Color Naming Models for Color Selection, Image Editing and Palette Design"
 */

 function getColorNameIndex(c) {
    var x = d3mlab(c),
        L = 5 * Math.round(x.L / 5),
        a = 5 * Math.round(x.a / 5),
        b = 5 * Math.round(x.b / 5),
        s = [L, a, b].join(",");
    return color_name_map[s];
}
function getColorSaliency(x) {
    let c = getColorNameIndex(x);
    return (c3.color.entropy(c) - minE) / (maxE - minE);
}
function getNameDifference(x1, x2) {
    let c1 = getColorNameIndex(x1),
        c2 = getColorNameIndex(x2);
    return 1 - c3.color.cosine(c1, c2);
}

/**
 * judge palette harmonious
 */
function judgeHarmonious(palette) {
    // let hues = palette.map(d => d3.hcl(d).h);
    let hues = null;
    if (harmony_mode === 'matsuda') {
        hues = palette.map(d => d3.hsl(d).h);
    } else if (harmony_mode === 'geometric') {
        hues = palette.map(d => d3.hcl(d).h);
    }
    let hue_res = harmony_judger.check_templates(hues);

    if (harmony_mode === 'geometric') {
        let lc_res = geo_lc_linear(palette);
        // console.log("harmonious", hue_res[0], lc_res);
        // hue_res[0] = lc_res;
        hue_res[0] += lc_res;
    }
    hue_res[0] *= 2 * Math.PI / 360; // convert to radian
    return hue_res;
}

/**
 * judge include target color
 */
function judgeInTargetColor(palette) {
    let count = 0;
    for (let i = 0; i < palette.length; i++) {
        let dist = d3_ciede2000(d3mlab(d3.rgb(palette[i])), best_color_obj);
        // console.log("dist", dist);
        if (dist > best_color_dis) {
            count++;
        }
    }
    if (count > best_count) {
        return false;
    }
    return true;
}


function normRGB(palette) {
    return palette.map(color => d3.rgb(norm255(color.r), norm255(color.g), norm255(color.b)));
}

/**
 * score the given palette
 */
function evaluatePalette(palette) {
    // palette = normRGB(palette);
    if (!judgeInTargetColor(palette)) {
        return -1000;
    }

    // calcualte data driven score
    let data_driven_value = use_data_driven_sa? data_driven_score(palette): 0;
    
    // calcualte color distance of given palette
    let name_difference = 0,
        color_discrimination_constraint = 100000;
    let dis;
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
            let nd = getNameDifference(palette[i], palette[j]);
            name_difference += nd;   
            color_discrimination_constraint = (color_discrimination_constraint > dis) ? dis : color_discrimination_constraint;
        }
        // dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(d3.rgb(bgcolor)));
        // color_discrimination_constraint = (color_discrimination_constraint > dis) ? dis : color_discrimination_constraint;
    }
    name_difference /= palette.length * (palette.length - 1) * 0.25;

    // calcualte color harmony of given palette
    let dist = judgeHarmonious(palette)[0];
    let harmony_value = -1 * dist;
    // console.log("harmony_value", harmony_value);
    // harmony_value = 0;

    let value = score_importance_weight[0] * data_driven_value + 
                score_importance_weight[1] * name_difference +
                score_importance_weight[2] * color_discrimination_constraint +
                score_importance_weight[3] * harmony_value;
    // console.log("score", score_importance_weight[0] * data_driven_value, name_difference, color_discrimination_constraint, score_importance_weight[3] * harmony_value);
    return value;
}

function genColorScope(color_hex, use_base_color = false) {
    let hcl = d3.hcl(color_hex);
    let scope = {};
    scope.hue_scope = [0, 360];
    scope.chroma_scope = [20, 100];
    scope.lumi_scope = [15, 85];

    if (use_base_color) {
        let l_divide = 60, l_range = 40;
        if (hcl.l > l_divide) {
            scope.lumi_scope = [hcl.l - l_range, hcl.l];
            if (scope.lumi_scope[0] < l_divide) {
                scope.lumi_scope[0] = l_divide;
            }
        }
        else {
            scope.lumi_scope = [hcl.l, hcl.l + l_range];
        }

        let c_range = 25;
        scope.chroma_scope = [hcl.c - c_range, hcl.c + c_range];
        if (scope.chroma_scope[0] < 10) {
            scope.chroma_scope[0] = 10;
        }
    }
    return scope;
}

/**
 * using simulated annealing to find the best palette of given data
 * @param {*} palette_size 
 * @param {*} evaluateFunc 
 * @param {*} colors_scope: hue range, lightness range, saturation range
 * @param {*} flag 
 */
function simulatedAnnealing2FindBestPalette(color_hex, palette_size, exclude_colors, colors_scope = { 'hue_scope': [0, 360], 'chroma_scope': [20, 100], 'lumi_scope': [15, 85] },
        data = null, labels = null) {
    if (palette_size == 1) {
        let color_palette = [d3.rgb(color_hex)];
        let preferredObj = {
            id: color_palette,
            score: 0
        };
        return preferredObj;
    }
    // adaptive simulated annealing args: iter max count, scope
    base_rgb = d3.rgb(color_hex);
    best_color_obj = d3mlab(d3.rgb(color_hex));
    best_color_dis = Math.min(palette_size * 4, 60);
    if (exclude_colors.length > 0) {
        let min_dist = 1e5;
        exclude_colors.forEach(color => {
            let dist = d3_ciede2000(d3mlab(color), best_color_obj);
            if (dist < min_dist) {
                min_dist = dist;
            }
        });
        best_color_dis = Math.max((min_dist - exclude_margin) / 2, min_best_color_dis);
        console.log('best_color_dis', best_color_dis);
    }

    // if color number is few, use less max count
    if(palette_size < 10) {
        let base_color_hcl = d3.hcl(color_hex);
        if (base_color_hcl.c < 20 || base_color_hcl.l < 20) {
            max_count = 50;
        }
        else max_count = 30;
    }
    else max_count = 100;
    // if color number is few, constrain scope for better performance
    // if (max_count < 40) colors_scope = genColorScope(color_hex, true);
    // else colors_scope = genColorScope(color_hex);

    // data driven 
    initDataDriven(data, labels);
    // console.log("ddi", data_driven_info)
    const evaluateFunc = evaluatePalette;
    

    let iterate_times = 0;
    //default parameters
    let max_temper = 100000,
        dec = global_dec, // 0.999
        max_iteration_times = 10000000,
        end_temper = 0.001;
    let cur_temper = max_temper;
    //generate a totally random palette
    let color_palette = getColorPaletteRandom(palette_size);
    let criterion_cd = -1.0;
    //evaluate the default palette
    let o = {
        id: color_palette,
        score: evaluateFunc(color_palette)
    },
    preferredObj = o;
    // return preferredObj;

    // let limit = 0;
    while (cur_temper > end_temper) {
        for (let i = 0; i < 1; i++) {//disturb at each temperature
            iterate_times++;
            color_palette = o.id.slice();  
            color_palette = disturbColors(color_palette);
            let color_palette_2 = normRGB(color_palette.slice());
            let o2 = {
                id: color_palette_2,
                score: evaluateFunc(color_palette_2)
            };
            // console.log('score', o2.score)
            
            // if (use_cross_opt) {
            //     o2.id = post_shift_palette(o2.id, colors_scope, cross_args[0], cross_args[1]);
            //     o2.score = evaluateFunc(o2.id);
            // }

            let delta_score = o.score - o2.score;
            if (delta_score <= 0 || delta_score > 0 && random() <= Math.exp((-delta_score) / cur_temper)) {
                o = o2;
                if (preferredObj.score - o.score < 0) {
                    preferredObj = o;
                }
            }
            if (iterate_times > max_iteration_times) { 
                break;
            }
        }

        cur_temper *= dec;
        // limit += 1;
        // if (limit > 0) {
        //     break;
        // }
    }
    if (data_driven_info['use'] && use_post) {
        preferredObj.id = post_shift_palette(preferredObj.id, 150 * palette_size, 20, true);
        preferredObj.score = evaluateFunc(preferredObj.id);
    } 

    // judge_harmony
    let res = judgeHarmonious(preferredObj.id);
    console.log('harmony', res);

    let dists = [];
    console.log('final score', preferredObj.score);
    for(let color of preferredObj.id) {
        dists.push(d3_ciede2000(d3mlab(color), best_color_obj));
        color.r = norm255(color.r);
        color.g = norm255(color.g);
        color.b = norm255(color.b);
    }
    console.log(dists);
    preferredObj.harmony_info = res;
    return preferredObj;
}

function post_shift_palette(palette, times=4000, limit_start=20, print=false) {
    let cur_shift_p = shift_p;
    let color_palette = palette.slice();
    let o = {
        id: palette,
        score: data_driven_score(color_palette)
    }
    shift_p = 0;
    if (print) console.log('start score part 0', o.score);
    // return o.id;

    let start_palettes = [];
    if (data_driven_info['type'] === 'discri') {
        start_palettes = [palette.slice()];
    } else {
        start_palettes = initShiftPosition(palette);
        if (start_palettes.length > limit_start) {
            // get random number of start palettes
            let start_palettes2 = [];
            let idxs = [];
            for (let i = 0; i < limit_start; i++) {
                let idx = getRandomIntInclusive(0, start_palettes.length - 1);
                while (idxs.includes(idx)) {
                    idx = getRandomIntInclusive(0, start_palettes.length - 1);
                }
                idxs.push(idx);
                start_palettes2.push(start_palettes[idx]);
            }
            start_palettes = start_palettes2;
        }
    }

    let each_times = Math.floor(times / start_palettes.length);

    start_palettes.forEach(palette_i => {
        let oi = {
            id: palette_i,
            score: data_driven_score(palette_i)
        };

        color_palette = palette_i.slice();
        for (let i = 0; i < each_times; i++) {
            color_palette = disturbPositionRandom(color_palette);
            color_palette = normRGB(color_palette);
            let o2 = {
                id: color_palette.slice(),
                score: data_driven_score(color_palette)
            };
    
            if (oi.score < o2.score) {
                oi = o2;
            }
        }
        if (o.score < oi.score) {
            o = oi;
        }
    })
    
    shift_p = cur_shift_p;
    if (print) console.log('final score part 2', o.score);
    return o.id;
}

// init shift position for close color position
function initShiftPosition(palette) {
    let labels = data_driven_info['labels'],
        label_centers = data_driven_info['label_centers'],
        center = data_driven_info['center'];
    let label_pos = label_centers.map((lcenter, idx) => {
        let dx = lcenter[0] - center[0],
            dy = lcenter[1] - center[1];
        let alpha = Math.atan2(dy, dx);
        return {
            label: labels[idx],
            alpha: alpha,
            id: idx,
        };
    });
    label_pos = label_pos.sort((a, b) => a.alpha - b.alpha);

    let palette_h = palette.slice();
    palette_h = palette_h.sort((a, b) => d3.hsl(a).h - d3.hsl(b).h);

    let palettes = [];
    let length = label_pos.length;
    for (let i = 0; i < length; i++) {
        let palette_i = palette.slice();
        let cur_i = i;
        for (let j = 0; j < label_pos.length; j++) {
            palette_i[label_pos[cur_i].id] = palette_h[j]
            cur_i = (cur_i + 1) % label_pos.length;
        }
        palettes.push(palette_i);
        break;
    }
    return palettes;
}


function getColorPaletteRandom(palette_size) {
    let palette = [];
    for (let i = 0; i < palette_size; i++) {
        let rgb = d3.rgb(getRandomIntInclusive(0, 255), getRandomIntInclusive(0, 255), getRandomIntInclusive(0, 255));
        palette.push(rgb);
    }
    return palette;
}

function randomDisturbColors(palette) {
    let disturb_step = 30;
    if (random_extend) disturb_step = 60;

    let idx = getRandomIntInclusive(0, palette.length - 1),
        color = glo_scope.disturbColor(palette[idx], getRandomIntInclusive, [disturb_step, disturb_step, disturb_step]);
    color = d3.rgb(color);
    
    palette[idx] = d3.rgb(norm255(color.r), norm255(color.g), norm255(color.b));
    let count = 0, sign;
    while (true) {
        while ((sign = isDiscriminative(palette)) > 0) {
            count += 1;
            if (count === max_count) {
                break;
            }
            color = glo_scope.disturbColor(palette[sign], getRandomIntInclusive, [disturb_step, disturb_step, disturb_step]);
            palette[sign] = d3.rgb(color);
        }

        let satisfy_inter_range = true;
        for (let i = 0; i < palette.length; i++) {
            let delta = 10;
            let dist = d3_ciede2000(d3mlab(d3.rgb(palette[i])), best_color_obj);
            if (dist > best_color_dis) {
                palette[i] = d3.rgb(glo_scope.disturbColor(base_rgb, getRandomIntInclusive, [delta, delta, delta]));
                satisfy_inter_range = false;
            }
        }

        if (satisfy_inter_range || count >= max_count) {
            break;
        }

    }
}


function isDiscriminative(palette) {
    let idx = -1;
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            let color_dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
            if (color_dis < global_color_dis) { 
                return j;
            }
        }
    }
    return idx;
}

function disturbPositionRandom(palette) {
    // randomly shuffle two colors of the palette 
    let idx_0 = getRandomIntInclusive(0, palette.length - 1),
    idx_1 = getRandomIntInclusive(0, palette.length - 1);
    while (idx_0 === idx_1) {
        idx_1 = getRandomIntInclusive(0, palette.length - 1);
    }
    let tmp = palette[idx_0];
    palette[idx_0] = palette[idx_1];
    palette[idx_1] = tmp;
    return palette;
}

/**
 * only use color discrimination
 * @param {} palette 
 */
function disturbColors(palette) {
    if (random() < shift_p) {
        randomDisturbColors(palette);
    } else {
        if (data_driven_info['type'] === 'similar') {
            palette = post_shift_palette(palette, cross_args[0], cross_args[1]);
        }
        else {
            palette = disturbPositionRandom(palette);
        }
        // disturbPositionRandom(palette);
    }
    return palette;
}
