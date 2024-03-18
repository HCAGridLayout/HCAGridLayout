/* eslint-disable */
import {matsuda_templates, geometric_hue_templates} from './hue_templates.js';
import geo_lc_linear from './geometric_lc.js';
export default findBestPaletteBetwClasses;
import ColorScope from './color_scope.js';
var glo_scope = new ColorScope();

// -------------------------------------------------------------------------------------------
/**
 * prepare color names and color table
 */
c3.load('./static/lib/c3_data.json');
var color_name_map = {};
for (var c = 0; c < c3.color.length; ++c) {
    var x = c3.color[c];
    color_name_map[[x.L, x.a, x.b].join(',')] = c;
}
var bgcolor = '#fff';


/**
 * prepare random seeds
 */
var seed = 45;
function random() {
    var x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
}


/**
 * arguments for color simulation
 */
// init colors save
var init_colors_save = [];
var init_color_dist = 20;
var init_delta = [10, 30, 30]; // h, c, l
var init_limit = 1;

// simulate annealing parameters
var global_dec = 0.99; // 0.99: high efficiency   0.999: high qualit
var max_count = 30;
var shift_p = 1;

// harmony judger
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

// data driven
var data_driven_info = {};
var data_driven_score = null;
var neighbor_num = 8; // 4 or 8
var similarity_type = 1; // 0: in class, 1: in neighbor class

// color evaluation
var global_color_dis = 20; // min dis between two color, default: 10
var score_importance_weight = [1  , // data driven: color discrimination / class similarity
    1,  // color name difference
    1,  // color discrimination constraint
    2];  // color harmony


// -------------------------------------------------------------------------------------------

/**
 * functions for data driven
 */
function initDataDriven(data, labels) { 
    // console.log(data, labels);
    // console.log('init data driven');
    // data_driven_info['use'] = false; return;

    data_driven_info = {};  
    if (data == null || labels == null || score_importance_weight[0] === 0) {
        data_driven_info['use'] = false;
        data_driven_score = function() { return 0; }
        return;
    }
    data_driven_info['use'] = true;
    data_driven_info['type'] = data['type'];
    data_driven_info['labels'] = labels;
    let label_map = {};
    let label_list = [];
    labels.forEach((clabels, idx) => {
        clabels.forEach(label => {
            label_map[label] = idx;
            label_list.push(label);
        });
    });
    let label_set = new Set(label_list);
    data_driven_info['set'] = label_set;
    data_driven_info['data'] = data['grids'].filter(grid => label_set.has(grid.label));
    data_driven_info['label_map'] = label_map;

    // console.error('data driven', data, labels);
    if (data['type'] === 'discri') {
        data_driven_score = data_driven_discri;
        data_driven_info['args'] = init_discri(data_driven_info['data'] , labels, label_map, label_set);
    } else {
        let bfmap = data['meta']['label_map'];
        let bfdists = data['meta']['label_dists'];
        let label_dists = labels.map(() => labels.map(() => 0));
        for (let i = 0; i < labels.length; i++) {
            for (let j = i + 1; j < labels.length; j++) {
                let dis = 0;
                for (let u = 0; u < labels[i].length; u++) {
                    for (let v = 0; v < labels[j].length; v++) {
                        dis += bfdists[bfmap[labels[i][u]]][bfmap[labels[j][v]]];
                    }
                }
                label_dists[i][j] = label_dists[j][i] = dis / (labels[i].length * labels[j].length);
            }
        }
        data_driven_info['label_dists'] = label_dists;

        data_driven_info['label_centers'] = labels.map(clabels => {
            let clset = new Set(clabels);
            let label_grids = data['grids'].filter(grid => clset.has(grid.label));
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
            data_driven_info['args'] = init_neigh_similar(data_driven_info['data'], labels, data_driven_info['label_map'], label_set, data_driven_info['label_dists']);
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
    let label_map = data_driven_info['label_map'];
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

function init_neigh_similar(grids, labels, label_map, label_set, dist_matrix, func = g) {
    let args = labels.map(() => labels.map(() => 0));
    grids.forEach((grid, idx) => {
        let n = 0;
        let label_id = label_map[grid.label];
        let res_labels = [];
        let res_dist = [];
        let judgelabel = function (label) {
            if (label_set.has(label) && label_map[label] !== label_id) {
                res_labels.push(label_map[label]);
                res_dist.push(func(1 - dist_matrix[label_map[label]][label_id]));
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
    // let labels = data_driven_info['labels'];
    // let label_map = data_driven_info['label_map'];
    let dis = 0;
    // from args calculate
    for (let i = 0; i < palette.length; i++) {
        for (let j = i + 1; j < palette.length; j++) {
            dis += d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j])) * args[i][j];
        }
    }
    return dis;
}

// -------------------------------------------------------------------------------------------

/**
 * calculating the Color Saliency 
 * reference to 'Color Naming Models for Color Selection, Image Editing and Palette Design'
 */

function getColorNameIndex(c) {
    var x = d3mlab(c),
        L = 5 * Math.round(x.L / 5),
        a = 5 * Math.round(x.a / 5),
        b = 5 * Math.round(x.b / 5),
        s = [L, a, b].join(',');
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
        // console.log('harmonious', hue_res[0], lc_res);
        // hue_res[0] = lc_res;
        hue_res[0] += lc_res;
    }
    hue_res[0] *= 2 * Math.PI / 360; // convert to radian
    return hue_res;
}

/**
 * judge include target color
 */
function judgeColorsInRange(palette) {
    let count = 0;
    for (let i = 0; i < palette.length; i++) {
        let dist = d3_ciede2000(d3mlab(palette[i]), init_colors_save[i].obj);
        if (dist > init_color_dist) {
            count++;
        }
    }
    if (count > init_limit) {
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
    if (!judgeColorsInRange(palette)) {
        return -1000;
    }

    // calcualte data driven score
    let data_driven_value = data_driven_info['use'] ? data_driven_score(palette): 0;
    
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
    }
    name_difference /= palette.length * (palette.length - 1) * 0.25;

    // calcualte color harmony of given palette
    let dist = judgeHarmonious(palette)[0];
    let harmony_value = -1 * dist;

    let value = score_importance_weight[0] * data_driven_value + 
                score_importance_weight[1] * name_difference +
                score_importance_weight[2] * color_discrimination_constraint +
                score_importance_weight[3] * harmony_value;
    // console.log('score', score_importance_weight[0] * data_driven_value, name_difference, color_discrimination_constraint, score_importance_weight[3] * harmony_value);
    return value;
}


// -------------------------------------------------------------------------------------------
/**
 * using simulated annealing to find the best palette of given data
 * @param {*} init_colors
 * @param {*} colors_scope
 * @param {*} data  # all backend data
 * @param {*} labels # all labels [[],[]] for each partitions
 */
function findBestPaletteBetwClasses(init_colors, colors_scope = { 'hue_scope': [0, 360], 'chroma_scope': [20, 100], 'lumi_scope': [15, 85] },
        data = null, labels = null) {
    // 1. init basic color parameters
    let palette_size = init_colors.length;
    if (palette_size == 1) {
        let color_palette = [d3.rgb(init_colors[0])];
        let preferredObj = {
            id: color_palette,
            score: 0
        };
        return preferredObj;
    }
    const evaluateFunc = evaluatePalette;
    init_colors_save = init_colors.map(color => {
        return {
            rgb: d3.rgb(color),
            hcl: d3.hcl(color),
            hsl: d3.hsl(color),
            raw: color,
            obj: d3mlab(d3.rgb(color))
        }
    });

    // data driven 
    // console.log('data', data, labels);
    initDataDriven(data, labels);

    // 2. init simulated annealing parameters
    let iterate_times = 0;
    let max_temper = 100000,
        dec = global_dec, 
        max_iteration_times = 10000000,
        end_temper = 0.001;
    let cur_temper = max_temper;
    let color_palette = init_colors_save.map(color => d3.rgb(color.raw));
    let o = {
        id: color_palette,
        score: evaluateFunc(color_palette)
    },
    preferredObj = o;
    // return preferredObj;

    // 3. simulated annealing
    while (cur_temper > end_temper) {
        for (let i = 0; i < 1; i++) {
            iterate_times++;
            color_palette = o.id.slice();  
            disturbColors(color_palette, colors_scope);
            color_palette = normRGB(color_palette);
            let o2 = {
                id: color_palette,
                score: evaluateFunc(color_palette)
            };

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
    }

    // 4. add harmony info and get score
    let res = judgeHarmonious(preferredObj.id);
    console.log('harmony', res);
    preferredObj.harmony_info = res;

    console.log('final score', preferredObj.score);
    return preferredObj;
}

function randomDisturbColors(palette, colors_scope) {
    let disturb_step = 30;

    let idx = getRandomIntInclusive(0, palette.length - 1),
        color = glo_scope.disturbColor(palette[idx], getRandomIntInclusive, [disturb_step, disturb_step, disturb_step]);
    palette[idx] = d3.rgb(color);

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

        let satisfy_init_constraint = true;
        for (let i = 0; i < palette.length; i++) {
            let dist = d3_ciede2000(d3mlab(palette[i]), init_colors_save[i].obj);
            if (dist > init_color_dist) {
                let hcl = init_colors_save[i].hcl;
                palette[i] = d3.rgb(glo_scope.disturbColor(hcl, getRandomIntInclusive, init_delta));
                satisfy_init_constraint = false;
            }
        }

        if (satisfy_init_constraint || count >= max_count) {
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

/**
 * only use color discrimination
 * @param {} palette 
 * @param {*} colors_scope 
 */
function disturbColors(palette, colors_scope) {
    if (random() < shift_p) {
        randomDisturbColors(palette, colors_scope);
    } else {
        // randomly shuffle two colors of the palette 
        let idx_0 = getRandomIntInclusive(0, palette.length - 1),
            idx_1 = getRandomIntInclusive(0, palette.length - 1);
        while (idx_0 === idx_1) {
            idx_1 = getRandomIntInclusive(0, palette.length - 1);
        }
        let tmp = palette[idx_0];
        palette[idx_0] = palette[idx_1];
        palette[idx_1] = tmp;
    }
}
