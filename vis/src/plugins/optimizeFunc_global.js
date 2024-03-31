/* eslint-disable */
import { matsuda_templates, geometric_hue_templates } from "./hue_templates.js";
import geo_lc_linear from "./geometric_lc.js";
export default findBestPaletteGlobal;
import ColorScope from "./color_scope.js";
import MTLAdjustArgs from "./mtl_adjust_args.js";
var glo_scope = new ColorScope();
var mtl_adjustor = null;

// -------------------------------------------------------------------------------------------
/**
 * prepare color names and color table
 */
c3.load("./static/lib/c3_data.json");
var color_name_map = {};
for (var c = 0; c < c3.color.length; ++c) {
  var x = c3.color[c];
  color_name_map[[x.L, x.a, x.b].join(",")] = c;
}
var bgcolor = "#fff";

/**
 * prepare random seeds
 */
var seed = 45;
function random() {
  var x = Math.sin(seed++) * 10000;
  return x - Math.floor(x);
}
function randomInt(min, max) {
  return Math.floor(random() * (max - min)) + min;
}
function randomDouble(min, max) {
  return random() * (max - min) + min;
}

/**
 * arguments for color simulation
 */
// basic color info save
var basic_info = [];
var color_margin = 5;
var color_min_range = 10;
var color_max_range = 60; // 60
var range_limit = 0;
var disturb_step = [30, 60, 60];
var idx2info = {};

// simulate annealing parameters
var global_dec = 0.99; // 0.99: high efficiency   0.999: high qualit
var max_count = 30;
var shift_p = 1;
var init_method = "random"; // 'random' or 'blue_noise'
var nsatisfy_discri_constarin_num = 0;
var judge_line = 30;
var dec_step = {
  discri_constrain: 0.9,
  discri_arg: 1.2
};

// harmony judger
var harmony_judger_matsuda = new matsuda_templates();
var harmony_judger_geo_hue = new geometric_hue_templates();
var harmony_mode = "matsuda"; // 'matsuda';
var harmony_judger = null;
if (harmony_mode === "matsuda") {
  harmony_judger = harmony_judger_matsuda;
} else if (harmony_mode === "geometric") {
  harmony_judger = harmony_judger_geo_hue;
  // harmony_judger = harmony_judger_matsuda;
}

// data driven
var data_driven_infos = [];
var data_driven_type = null;
var use_data_driven_sa = true;
var cross_args = [1, 1]; // times and num [2, 2]
var data_driven_score = null;
var neighbor_num = 8; // 4 or 8
var similarity_type = 1; // 0: in class, 1: in neighbor class

// color evaluation
var global_color_dis = 10; // min dis between two color, default: 10 // 20
var score_importance_weight = [
  1, // data driven: color discrimination / class similarity
  1, // color name difference
  1, // color discrimination constraint
  1
]; // color harmony

// -------------------------------------------------------------------------------------------
/**
 * functions for data driven
 */
function initDataDriven(data, labels) {
  // console.log(data, labels);
  // console.log("init data driven");
  let data_driven_info = {};
  if (data == null || labels == null || score_importance_weight[0] === 0) {
    data_driven_info["use"] = false;
    // shift_p = 1;
    data_driven_score = function() {
      return 0;
    };
    return;
  }
  data_driven_info["use"] = true;
  // data_driven_info['type'] = data['type'];
  data_driven_info["labels"] = labels;
  let label_set = new Set(labels);
  data_driven_info["set"] = label_set;
  data_driven_info["data"] = data["grids"].filter(grid =>
    label_set.has(grid.label)
  );
  let label_map = {};
  labels.forEach((label, idx) => {
    label_map[label] = idx;
  });
  data_driven_info["label_map"] = label_map;
  // shift_p = 0.5;

  if (data["type"] === "discri") {
    // data_driven_score = data_driven_discri;
    data_driven_info["args"] = init_discri(
      data_driven_info["data"],
      labels,
      label_map,
      label_set
    );
  } else {
    data_driven_info["label_map2dist"] = data["meta"]["label_map"];
    data_driven_info["label_dists"] = data["meta"]["label_dists"];
    data_driven_info["label_centers"] = labels.map(label => {
      let label_grids = data["grids"].filter(grid => grid.label === label);
      let label_sum = label_grids.reduce(
        (sum, grid) => {
          sum[0] += grid.x;
          sum[1] += grid.y;
          return sum;
        },
        [0, 0]
      );
      return [
        label_sum[0] / label_grids.length + label_grids[0].width / 2,
        label_sum[1] / label_grids.length + label_grids[0].height / 2
      ];
    });
    data_driven_info["center"] = [
      data_driven_info["label_centers"].reduce(
        (sum, center) => sum + center[0],
        0
      ) / data_driven_info["label_centers"].length,
      data_driven_info["label_centers"].reduce(
        (sum, center) => sum + center[1],
        0
      ) / data_driven_info["label_centers"].length
    ];

    if (similarity_type === 1) {
      //     data_driven_score = data_driven_similarity;
      // } else {
      data_driven_info["args"] = init_neigh_similar(
        data_driven_info["data"],
        labels,
        data_driven_info["label_map"],
        data_driven_info["label_map2dist"],
        label_set,
        data_driven_info["label_dists"]
      );
      // data_driven_score = data_driven_similarity2;
    }
  }
  return data_driven_info;
}

function init_discri(grids, labels, label_map, label_set) {
  let args = labels.map(() => labels.map(() => 0));
  let num_sum = 0;
  grids.forEach((grid, idx) => {
    let n = 0;
    let label_id = label_map[grid.label];
    let res_labels = [];
    let res_dist = [];
    let judgelabel = function(label, dist) {
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
    num_sum += n;
  });
  let sum = 0;
  for (let i = 0; i < args.length; i++) {
    for (let j = i + 1; j < args.length; j++) {
      args[i][j] /= grids.length;
      args[j][i] /= grids.length;
      sum += args[i][j] * 2;
    }
  }
  // console.log(args, num_sum, sum);
  return args;
}

function data_driven_discri(palette) {
  let dis_sum = 0;
  basic_info.forEach((info, idx) => {
    let data_driven_info = data_driven_infos[idx];
    let args = data_driven_info["args"];
    let dis = 0;
    // from args calculate
    for (let i = 0; i < info.num; i++) {
      for (let j = i + 1; j < info.num; j++) {
        dis +=
          d3_ciede2000(
            d3mlab(palette[i + info.start]),
            d3mlab(palette[j + info.start])
          ) * args[i][j];
      }
    }
    dis_sum += dis * info.ratio;
  });
  return dis_sum;
}

var epsilon = 1e-4;
function g(similarity) {
  // change similarity from (-1,1) to (-inf, 0)
  return Math.log(1 - 0.5 * (1 + similarity));
}

function data_driven_similarity(palette, func = g) {
  let dis_sum = 0;
  basic_info.forEach((info, idx) => {
    let data_driven_info = data_driven_infos[idx];
    let labels = data_driven_info["labels"];
    let label_map = data_driven_info["label_map2dist"];
    let matrix = data_driven_info["label_dists"];
    let dis = 0;
    for (let i = 0; i < info.num; i++) {
      for (let j = i + 1; j < info.num; j++) {
        let color_dist = d3_ciede2000(
          d3mlab(palette[i + info.start]),
          d3mlab(palette[j + info.start])
        ); // Math.abs(palette[i].h - palette[j].h);
        let label_dist = matrix[label_map[labels[i]]][label_map[labels[j]]];
        // dis += (1 - label_dist) / color_dist;
        dis += func(1 - label_dist) * color_dist;
      }
    }
    // dis /= info.num;
    dis_sum += dis * info.ratio;
  });
  return dis_sum;
}

function init_neigh_similar(
  grids,
  labels,
  label_map,
  label_map2dist,
  label_set,
  dist_matrix,
  func = g
) {
  let args = labels.map(() => labels.map(() => 0));
  grids.forEach((grid, idx) => {
    let n = 0;
    let label_id = label_map[grid.label];
    let label_dist_id = label_map2dist[grid.label];
    let res_labels = [];
    let res_dist = [];
    let judgelabel = function(label) {
      if (label_set.has(label) && label_map[label] !== label_id) {
        res_labels.push(label_map[label]);
        res_dist.push(
          func(1 - dist_matrix[label_map2dist[label]][label_dist_id])
        );
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
  let sum = 0;
  for (let i = 0; i < args.length; i++) {
    for (let j = i + 1; j < args.length; j++) {
      args[i][j] /= grids.length;
      args[j][i] /= grids.length;
      sum += args[i][j] * 2;
    }
  }
  // console.log(args, sum);
  return args;
}

function data_driven_similarity2(palette) {
  let dis_sum = 0;
  basic_info.forEach((info, idx) => {
    let data_driven_info = data_driven_infos[idx];
    let args = data_driven_info["args"];
    let labels = data_driven_info["labels"];
    let label_map = data_driven_info["label_map"];
    let dis = 0;
    // from args calculate
    for (let i = 0; i < info.num; i++) {
      for (let j = i + 1; j < info.num; j++) {
        dis +=
          d3_ciede2000(
            d3mlab(palette[i + info.start]),
            d3mlab(palette[j + info.start])
          ) * args[label_map[labels[i]]][label_map[labels[j]]];
      }
    }
    dis_sum += dis * info.ratio;
  });
  return dis_sum;
}

// -------------------------------------------------------------------------------------------

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

function getNameDifference(x1, x2) {
  let c1 = getColorNameIndex(x1),
    c2 = getColorNameIndex(x2);
  return 1 - c3.color.cosine(c1, c2);
}

/**
 * judge palette harmonious
 */
function judgeHarmoniousTotal(palette) {
  let result = 0;
  basic_info.forEach(info => {
    let palette_slice = palette.slice(info.start, info.end);
    let res = judgeHarmonious(palette_slice);
    result += res[0] * info.ratio;
  });
  return result;
}

function judgeHarmonious(palette) {
  // let hues = palette.map(d => d3.hcl(d).h);
  let hues = null;
  if (harmony_mode === "matsuda") {
    hues = palette.map(d => d3.hsl(d).h);
  } else if (harmony_mode === "geometric") {
    hues = palette.map(d => d3.hcl(d).h);
  }
  let hue_res = harmony_judger.check_templates(hues);

  if (harmony_mode === "geometric") {
    let lc_res = geo_lc_linear(palette);
    // console.log("harmonious", hue_res[0], lc_res);
    // hue_res[0] = lc_res;
    hue_res[0] += lc_res;
  }
  hue_res[0] *= (2 * Math.PI) / 360; // convert to radian
  return hue_res;
}

/**
 * judge include target color
 */
function judgeColorsInRange(palette) {
  let count = 0;
  basic_info.forEach(info => {
    for (let i = info.start; i < info.end; i++) {
      let dis = d3_ciede2000(d3mlab(palette[i]), info.obj);
      if (dis > info.range) {
        count += 1;
        // console.log("out of range", dis, info.range, i);
      }
    }
  });
  if (count > range_limit) {
    return false;
  }
  return true;
}

function normRGB(palette) {
  return palette.map(color =>
    d3.rgb(norm255(color.r), norm255(color.g), norm255(color.b))
  );
}

function normrgb(rgb) {
  rgb.r = norm255(rgb.r);
  rgb.g = norm255(rgb.g);
  rgb.b = norm255(rgb.b);
  return rgb;
}

/**
 * score the given palette
 */
function evaluatePalette(palette) {
  // palette = normRGB(palette);
  // if (!judgeColorsInRange(palette)) {
  //     return -1000;
  // }

  // calcualte data driven score
  let data_driven_value = use_data_driven_sa ? data_driven_score(palette) : 0;
  // console.log("data_driven_value", data_driven_value);

  // calcualte color distance of given palette
  let name_difference = 0,
    color_discrimination_constraint = 100000;
  let dis;
  for (let i = 0; i < palette.length; i++) {
    for (let j = i + 1; j < palette.length; j++) {
      dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
      let nd = getNameDifference(palette[i], palette[j]);
      name_difference += nd;
      color_discrimination_constraint =
        color_discrimination_constraint > dis
          ? dis
          : color_discrimination_constraint;
    }
    // dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(d3.rgb(bgcolor)));
    // color_discrimination_constraint = (color_discrimination_constraint > dis) ? dis : color_discrimination_constraint;
  }
  name_difference /= palette.length * (palette.length - 1) * 0.25;

  min_dist = color_discrimination_constraint;
  if (color_discrimination_constraint < global_color_dis) {
    nsatisfy_discri_constarin_num += 1;
  } else {
    nsatisfy_discri_constarin_num = 0;
  }

  max_deltac = init_deltac.slice();
  basic_info.forEach((info, idx) => {
    // get dist to center
    let center = info.obj;
    for (let i = info.start; i < info.end; i++) {
      let tpdist = d3_ciede2000(d3mlab(palette[i]), center);
      if (tpdist > max_deltac[idx]) {
        max_deltac[idx] = tpdist;
      }
    }
  });

  // calcualte color harmony of given palette
  let dist = judgeHarmoniousTotal(palette);
  let harmony_value = -1 * dist;
  // console.log("harmony_value", harmony_value);
  // harmony_value = 0;

  let value =
    score_importance_weight[0] * data_driven_value +
    score_importance_weight[1] * name_difference +
    score_importance_weight[2] * color_discrimination_constraint +
    score_importance_weight[3] * harmony_value;
  // console.log(
  //   "score",
  //   data_driven_value,
  //   name_difference,
  //   color_discrimination_constraint,
  //   harmony_value
  // );
  return value;
}

// -------------------------------------------------------------------------------------------
/**
 * functions for calculate MTL loss
 */
var max_deltac = [];
var init_deltac = [];
var min_dist = 0;
// var loss1 = (mindelta) => Math.pow(Math.max(0, 10-mindelta), 2);
var loss1 = mindelta => Math.abs(Math.max(0, 10 - mindelta));
var loss2_generator = (Clengths, Crs) => {
  let w = Clengths.map(len => 1 / len);
  let wsum = w.reduce((sum, val) => sum + val);
  return maxdeltas => {
    let delta = maxdeltas.reduce((sum, maxdelta, i) => {
      return sum + w[i] * (maxdelta - Crs[i]);
    }, 0);
    // return Math.pow(Math.max(0, delta / wsum), 2);
    return Math.abs(Math.max(0, delta / wsum));
  };
};
var loss2 = null;
var relax_args = {
  dist: 0.9,
  range: 1.1
};
var penalize_alpha = 2;

// -------------------------------------------------------------------------------------------
/**
 * using simulated annealing to find the best palette of given data
 * @param {*} palette_size
 * @param {*} evaluateFunc
 * @param {*} colors_scope: hue range, lightness range, saturation range
 * @param {*} flag
 */
function findBestPaletteGlobal(
  basic_colors,
  palette_sizes,
  part_ratios = null,
  data = null,
  labels_list = null
) {
  // 1. init basic color info and distance
  if (part_ratios == null) {
    let total_size = palette_sizes.reduce((sum, size) => sum + size, 0);
    part_ratios = palette_sizes.map(size => size / total_size);
  }
  console.assert(basic_colors.length === palette_sizes.length);
  console.assert(basic_colors.length === part_ratios.length);

  let start_idx = 0,
    end_idx = 0;
  basic_info = basic_colors.map((color, idx) => {
    start_idx = end_idx;
    end_idx += palette_sizes[idx];
    return {
      id: idx,
      start: start_idx,
      end: end_idx,
      rgb: d3.rgb(color),
      hcl: d3.hcl(color),
      num: palette_sizes[idx],
      ratio: part_ratios[idx],
      obj: d3mlab(color)
      // data driven
    };
  });
  let dist_matrix = basic_info.map(info =>
    basic_info.map(info2 => d3_ciede2000(info.obj, info2.obj))
  );
  // let h_dist_matrix = basic_info.map(info => basic_info.map(info2 => Math.abs(info.hcl.h - info2.hcl.h)));
  let sqnums = basic_info.map(info => Math.sqrt(info.num));
  basic_info.forEach((info, i) => {
    info["range"] = Math.max(
      dist_matrix[info.id].reduce((min, val, j) => {
        if (val > 0) {
          let r = ((val - color_margin) * sqnums[i]) / (sqnums[i] + sqnums[j]);
          if (r < min) return r;
        }
        return min;
      }, 1e5),
      color_min_range
    );
    if (info["range"] > color_max_range) {
      info["range"] = color_max_range;
      global_color_dis = 20; // enough space and set better standard
    }
  });
  basic_info.forEach((info, idx) => {
    for (let i = info.start; i < info.end; i++) {
      idx2info[i] = idx;
    }
  });
  // console.log("basic_info", basic_info);
  // data driven
  // initDataDriven(data, labels);
  shift_p = 1;
  if (data != null && labels_list != null) {
    data_driven_infos = labels_list.map(labels => {
      return initDataDriven(data, labels);
    });
    data_driven_type = data["type"];
    if (data_driven_type === "discri") {
      data_driven_score = data_driven_discri;
    } else {
      if (similarity_type === 0) {
        data_driven_score = data_driven_similarity;
      } else {
        data_driven_score = data_driven_similarity2;
      }
    }
    shift_p = 0.5;
  }
  const evaluateFunc = evaluatePalette;
  loss2 = loss2_generator(
    basic_info.map(info => info.num),
    basic_info.map(info => info.range)
  );
  init_deltac = basic_info.map(info => info.range);
  // console.log("init_deltac", init_deltac);
  let mtl_adjustor = new MTLAdjustArgs(loss1, loss2);

  // 2. init simulated annealing parameters
  let iterate_times = 0;
  let max_temper = 100000,
    dec = global_dec, // 0.999
    max_iteration_times = 10000000,
    end_temper = 0.001;
  let cur_temper = max_temper;

  let color_palette =
    init_method === "random"
      ? getColorPaletteRandom()
      : getColorPaletteBlueNoise();
  let o = {
      id: color_palette,
      score: evaluateFunc(color_palette),
      loss: mtl_adjustor.getloss([min_dist], [max_deltac])
    },
    preferredObj = o;
  console.log("start score", o.score);
  nsatisfy_discri_constarin_num = 0;
  // return preferredObj;
  let soft_constrain = false;
  let need_relax = false;
  let loss_delta = 0;

  while (cur_temper > end_temper) {
    // for (let i = 0; i < 1; i++) {//disturb at each temperature
    iterate_times++;
    color_palette = o.id.slice();
    color_palette = disturbColors(color_palette);
    // color_palette = normRGB(color_palette);
    let o2 = {
      id: color_palette,
      score: evaluateFunc(color_palette),
      loss: mtl_adjustor.getloss([min_dist], [max_deltac])
    };
    // console.log('score', o, o2, o2.score);

    // // judge temper
    // if (nsatisfy_discri_constarin_num == judge_line) {
    //     global_color_dis *= dec_step['discri_constrain'];
    //     score_importance_weight[2] *= dec_step['discri_arg'];
    //     cur_temper = max_temper;
    //     o.score = evaluateFunc(o.id);
    //     o2.score = evaluateFunc(o2.id);
    //     console.log('reset temper', global_color_dis, score_importance_weight[2], o.score, o2.score);
    //     nsatisfy_discri_constarin_num = 0;
    //     soft_constrain = true;
    // }

    if (nsatisfy_discri_constarin_num == judge_line) {
      soft_constrain = true;
      need_relax = true;
      cur_temper = max_temper;
      nsatisfy_discri_constarin_num = 0;
    }
    if (soft_constrain) {
      let relax_id = mtl_adjustor.adjust(o2.loss, need_relax);
      if (need_relax) {
        if (relax_id === 0) {
          global_color_dis *= relax_args["dist"];
          console.log("relax dist", global_color_dis);
        } else {
          basic_info.forEach(info => {
            info["range"] *= relax_args["range"];
          });
          console.log(
            "relax range",
            basic_info.map(info => info.range)
          );
        }
        need_relax = false;
      }
      loss_delta =
        (mtl_adjustor.getvalueCur(o.loss) - mtl_adjustor.getvalueCur(o2.loss)) *
        penalize_alpha;
      // console.log('soft', o.score, o2.score, o.loss, o2.loss,
      // o.score-mtl_adjustor.getvalueCur(o.loss), o2.score-mtl_adjustor.getvalueCur(o2.loss));
      // console.log(o.loss, o2.loss, mtl_adjustor.argsHistory);
      // break;
    }

    let delta_score = o.score - o2.score - loss_delta;
    if (
      delta_score <= 0 ||
      (delta_score > 0 && random() <= Math.exp(-delta_score / cur_temper))
    ) {
      o = o2;
      if (preferredObj.score - o.score < 0) {
        preferredObj = o;
      }
    }
    if (iterate_times > max_iteration_times) {
      break;
    }
    // }

    cur_temper *= dec;
  }
  // if (data_driven_info['use'] && use_post) {
  //     preferredObj.id = post_shift_palette(preferredObj.id, 150 * palette_size, 20, true);
  //     preferredObj.score = evaluateFunc(preferredObj.id);
  // }

  // judge_harmony
  console.log(mtl_adjustor);
  let res = judgeHarmoniousTotal(preferredObj.id);
  console.log("harmony", res);
  console.log("final score", preferredObj.score);
  return preferredObj;
}

function getColorPaletteRandom() {
  let palette = [];
  basic_info.forEach(info => {
    d3.range(info.num).forEach(idx => {
      let color = glo_scope.disturbColor(info.hcl, randomDouble, [
        info.range,
        60,
        60
      ]);
      palette.push(modifyInRange(color, idx + info.start));
    });
  });
  // console.log(normRGB(palette));
  return palette;
}

function getColorPaletteBlueNoise() {
  let palette = [];
  let objs = [];
  basic_info.forEach(info => {
    d3.range(info.num).forEach(idx => {
      // console.log(idx);
      idx = idx + info.start;
      let color = glo_scope.disturbColor(info.hcl, randomDouble, [
        info.range,
        60,
        60
      ]);
      color = modifyInRange(color, idx);
      let obj = d3mlab(color);

      let count = 0,
        flag = false;
      while (count <= max_count) {
        flag = true;
        for (let i = 0; i < palette.length; i++) {
          let color_dis = d3_ciede2000(objs[i], obj);
          if (color_dis < global_color_dis) {
            flag = false;
            break;
          }
        }
        if (flag) {
          break;
        }
        color = glo_scope.disturbColor(info.hcl, randomDouble, [
          info.range,
          60,
          60
        ]);
        color = modifyInRange(color, idx);
        obj = d3mlab(color);
        count += 1;
      }
      objs.push(obj);
      palette.push(color);
    });
  });
  return palette;
}

function modifyInRange(color, idx) {
  let binfo = basic_info[idx2info[idx]];
  color = d3.hcl(color);
  let rgb = normrgb(d3.rgb(color));
  let dis = d3_ciede2000(d3mlab(rgb), binfo.obj);
  let ratio = dis > 0 ? binfo.range / dis : 1;
  while (dis > binfo.range) {
    color = d3.hcl(
      color.h - (color.h - binfo.hcl.h) * ratio,
      color.c - (color.c - binfo.hcl.c) * ratio,
      color.l - (color.l - binfo.hcl.l) * ratio
    );
    rgb = normrgb(d3.rgb(color));
    dis = d3_ciede2000(d3mlab(rgb), binfo.obj);
  }
  return rgb;
}

function randomDisturbColors(palette) {
  let idx = getRandomIntInclusive(0, palette.length - 1),
    color = glo_scope.disturbColor(palette[idx], randomDouble, disturb_step);
  palette[idx] = modifyInRange(color, idx);

  let count = 0,
    sign;
  // while (true) {
  while ((sign = isDiscriminative(palette)) > 0) {
    count += 1;
    if (count === max_count) {
      // console.log('error', sign);
      break;
    }
    color = glo_scope.disturbColor(palette[sign], randomDouble, disturb_step);
    palette[sign] = modifyInRange(color, sign);
  }

  // let satisfy_inter_range = true;
  // basic_info.forEach(info => {
  //     for (let i = info.start; i < info.end; i++) {
  //         let dis = d3_ciede2000(d3mlab(palette[i]), info.obj);
  //         if (dis > info.range) {
  //             satisfy_inter_range = false;
  //         }
  //         if (dis > info.range) {
  //             let cur = d3.hcl(palette[i]);
  //             palette[i] = d3.rgb(d3.hcl(info.hcl.h - (info.hcl.h - cur.h) * info.range / dis,
  //                     info.hcl.c - (info.hcl.c - cur.c) * info.range / dis,
  //                     info.hcl.l - (info.hcl.l - cur.l) * info.range / dis));
  //             // dis = d3_ciede2000(d3mlab(palette[i]), info.obj);
  //         }
  //     }
  // });

  // if (satisfy_inter_range || count >= max_count) {
  //     break;
  // }
  // }
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
  let palette_part = randomInt(0, basic_info.length - 1);
  let info = basic_info[palette_part];

  let idx_0 = randomInt(0, info.num - 1),
    idx_1 = randomInt(0, info.num - 1);
  while (idx_0 === idx_1) {
    idx_1 = randomInt(0, info.num - 1);
  }
  idx_0 += info.start;
  idx_1 += info.start;
  let tmp = palette[idx_0];
  palette[idx_0] = palette[idx_1];
  palette[idx_1] = tmp;
  return palette;
}

function disturbPositionHue(palette, times = 4000, limit_start = 20) {
  let cur_shift_p = shift_p;
  let color_palette = palette.slice();
  let o = {
    id: palette,
    score: data_driven_score(color_palette)
  };
  shift_p = 0;

  let start_palettes = initShiftPosition(palette, limit_start);
  let each_times = Math.floor(times / start_palettes.length);

  start_palettes.forEach(palette_i => {
    let oi = {
      id: palette_i,
      score: data_driven_score(palette_i)
    };

    color_palette = palette_i.slice();
    for (let i = 0; i < each_times; i++) {
      color_palette = disturbPositionRandom(color_palette);
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
  });
  shift_p = cur_shift_p;
  return o.id;
}

// init shift position for close color position
function initShiftPosition(palette, limit_start = 20) {
  let palette_choices = [];
  basic_info.forEach((info, idx) => {
    let data_driven_info = data_driven_infos[idx];
    let labels = data_driven_info["labels"],
      label_centers = data_driven_info["label_centers"],
      center = data_driven_info["center"];
    let label_pos = label_centers.map((lcenter, idx) => {
      let dx = lcenter[0] - center[0],
        dy = lcenter[1] - center[1];
      let alpha = Math.atan2(dy, dx);
      return {
        label: labels[idx],
        alpha: alpha,
        id: idx
      };
    });
    label_pos = label_pos.sort((a, b) => a.alpha - b.alpha);

    let palette_h = palette.slice(info.start, info.end);
    palette_h = palette_h.sort((a, b) => d3.hsl(a).h - d3.hsl(b).h);

    let palettes = [];
    let length = label_pos.length;
    for (let i = 0; i < length; i++) {
      let palette_i = palette_h.slice();
      let cur_i = i;
      for (let j = 0; j < label_pos.length; j++) {
        palette_i[label_pos[cur_i].id] = palette_h[j];
        cur_i = (cur_i + 1) % label_pos.length;
      }
      palettes.push(palette_i);
      break;
    }
    palette_choices.push(palettes);
  });

  let start_palettes = [];
  for (let i = 0; i < limit_start; i++) {
    let palette = [];
    palette_choices.forEach(palettes => {
      let idx = randomInt(0, palettes.length - 1);
      palette = palette.concat(palettes[idx]);
    });
    start_palettes.push(palette);
  }
  return start_palettes;
}

/**
 * only use color discrimination
 * @param {} palette
 */
function disturbColors(palette) {
  if (random() < shift_p) {
    randomDisturbColors(palette);
  } else {
    if (data_driven_type === "similar") {
      palette = disturbPositionHue(palette, cross_args[0], cross_args[1]);
    } else {
      palette = disturbPositionRandom(palette);
    }
    // disturbPositionRandom(palette);
  }
  return palette;
}
