/* eslint-disable */
import * as d3 from "d3";
import { matsuda_templates, geometric_hue_templates } from "./hue_templates.js";
import * as kdtree from "static-kdtree";
import getPreferenceScore from "./pp_score.js";

// ------------ prepare for color distance--------------------------------
let c3 = {};
/**
 * calculating the Color Saliency
 * reference to "Color Naming Models for Color Selection, Image Editing and Palette Design"
 */
function c3_init(json) {
  var i, C, W, T, A, ccount, tcount;

  // parse colors
  c3.color = [];
  for (i = 0; i < json.color.length; i += 3) {
    c3.color[i / 3] = d3.lab(
      json.color[i],
      json.color[i + 1],
      json.color[i + 2]
    );
  }
  C = c3.color.length;

  // parse terms
  c3.terms = json.terms;
  W = c3.terms.length;

  // parse count table
  c3.T = T = [];
  for (var i = 0; i < json.T.length; i += 2) {
    T[json.T[i]] = json.T[i + 1];
  }

  // construct counts
  c3.color.count = ccount = [];
  for (i = 0; i < C; ++i) ccount[i] = 0;
  c3.terms.count = tcount = [];
  for (i = 0; i < W; ++i) tcount[i] = 0;
  d3.range(T.length).forEach(function(idx) {
    var c = Math.floor(idx / W),
      w = Math.floor(idx % W),
      v = T[idx] || 0;
    ccount[c] += v;
    tcount[w] += v;
  });

  // parse word association matrix
  c3.A = A = json.A;
}
function c3_api() {
  var C = c3.color.length,
    W = c3.terms.length,
    T = c3.T,
    A = c3.A,
    ccount = c3.color.count,
    tcount = c3.terms.count;

  c3.color.cosine = function(a, b) {
    var sa = 0,
      sb = 0,
      sc = 0,
      ta,
      tb;
    for (var w = 0; w < W; ++w) {
      ta = T[a * W + w] || 0;
      tb = T[b * W + w] || 0;
      sa += ta * ta;
      sb += tb * tb;
      sc += ta * tb;
    }
    return sc / Math.sqrt(sa * sb);
  };

  c3.color.vector = function(a) {
    var v = [];
    for (var w = 0; w < W; ++w) {
      v.push(T[a * W + w] || 0);
    }
    return v;
  };
}
c3 = { version: "1.0.0" };
c3.load = function(uri, async) {
  async = async || false;
  var req = new XMLHttpRequest();
  var onload = function() {
    if (!async || req.readyState == 4) {
      if (req.status == 200 || req.status == 0) {
        c3_init(JSON.parse(req.responseText));
        c3_api();
      } else {
        alert("Error Loading C3 Data");
      }
    }
  };
  req.open("GET", uri, false);
  if (async) req.onreadystatechange = onload;
  req.send(null);
  if (!async) onload();
};
c3.load("./static/lib/c3_data.json");
// color name lookup table
let color_name_map = {};
let kd_points = [];
for (var c = 0; c < c3.color.length; ++c) {
  var x = c3.color[c];
  kd_points.push([x.l, x.a, x.b]);
  color_name_map[[x.l, x.a, x.b].join(",")] = c;
}

// kd tree for color name lookup
let kd = kdtree(kd_points);

function getColorNameIndex(c) {
  var x = d3mlab(c),
    // L = 5 * Math.round(x.L / 5),
    // a = 5 * Math.round(x.a / 5),
    // b = 5 * Math.round(x.b / 5),
    // s = [L, a, b].join(",");
    index = kd.nn([x.L, x.a, x.b]),
    s = kd_points[index].join(",");
  // console.log([x.L, x.a, x.b], s, index);
  return color_name_map[s];
}
function getNameDifference(x1, x2) {
  let c1 = getColorNameIndex(x1),
    c2 = getColorNameIndex(x2);
  // console.log(1 - c3.color.cosine(c1, c2));
  return 1 - c3.color.cosine(c1, c2);
}

function H(vector) {
  let sum = 0;
  for (let i = 0; i < vector.length; i++) {
    if (vector[i] === 0) continue;
    sum += vector[i] * Math.log(vector[i]);
  }
  return -sum;
}

function getNameUnique(x) {
  let c = getColorNameIndex(x);
  //   console.log(c, c3.color.vector(c));
  return -H(c3.color.vector(c));
}

// ----------- evaluate the colors distance ---------------------
function judge_color_dist(palette) {
  let res = {};
  let min_perception_difference = 100000,
    perception_difference = 0,
    min_name_difference = 100000,
    name_difference = 0,
    total_num = 0,
    name_unique = 0;
  for (let i = 0; i < palette.length; i++) {
    for (let j = i + 1; j < palette.length; j++) {
      let dis = d3_ciede2000(d3mlab(palette[i]), d3mlab(palette[j]));
      let nd = getNameDifference(palette[i], palette[j]);
      perception_difference += dis;
      name_difference += nd;
      total_num++;
      min_perception_difference =
        min_perception_difference > dis ? dis : min_perception_difference;
      min_name_difference = min_name_difference > nd ? nd : min_name_difference;
    }
    name_unique += getNameUnique(palette[i]);
  }
  if (total_num > 0) {
    name_difference /= total_num;
    perception_difference /= total_num;
  }
  name_unique /= palette.length;
  res["perception_difference"] = perception_difference;
  res["min_perception_difference"] = min_perception_difference;
  res["name_difference"] = name_difference;
  res["min_name_difference"] = min_name_difference;
  res["name_unique"] = name_unique;
  res["number"] = palette.length;
  res["number_quad"] = total_num;
  return res;
}

function summary_color_dist(palettes) {
  let total_res = {};
  let min_perception_difference = 100000,
    perception_difference = 0,
    min_name_difference = 100000,
    name_difference = 0,
    total_num_quad = 0,
    total_num = 0,
    name_unique = 0,
    bi_total_num = 0;

  palettes.forEach(palette => {
    let res = judge_color_dist(palette);
    if (res["number_quad"] > 0) {
      perception_difference += res["perception_difference"] * res["number"];
      name_difference += res["name_difference"] * res["number"];
      bi_total_num += res["number"];
    }
    total_num_quad += res["number_quad"];
    min_perception_difference =
      min_perception_difference > res["min_perception_difference"]
        ? res["min_perception_difference"]
        : min_perception_difference;
    min_name_difference =
      min_name_difference > res["min_name_difference"]
        ? res["min_name_difference"]
        : min_name_difference;
    name_unique += res["name_unique"] * res["number"];
    total_num += res["number"];
  });

  if (bi_total_num === 0) {
    min_perception_difference = 100000;
    min_name_difference = 100000;
    name_difference = 100000;
    perception_difference = 100000;
  } else {
    name_difference /= bi_total_num;
    perception_difference /= bi_total_num;
  }

  name_unique /= total_num;
  total_res["perception_difference"] = perception_difference;
  total_res["min_perception_difference"] = min_perception_difference;
  total_res["name_difference"] = name_difference;
  total_res["min_name_difference"] = min_name_difference;
  total_res["name_unique"] = name_unique;
  total_res["number"] = total_num;
  total_res["number_quad"] = total_num_quad;
  return total_res;
}

// ----------- evaluate the colors harmony/preference ---------------------
let matsuda = new matsuda_templates();
let geometric = new geometric_hue_templates();

const angle2rad = Math.PI / 180;
function E_SY(h, c, l) {
  let Ec = 0.5 + 0.5 * Math.tanh(-2 + 0.5 * c);
  let Hs =
    -0.08 -
    0.14 * Math.sin((h + 50) * angle2rad) -
    0.07 * Math.sin((2 * h + 90) * angle2rad);
  let tp = (90 - h) / 10;
  let EY = ((0.22 * l - 12.8) / 10) * Math.exp(tp - Math.exp(tp));
  return Ec * (Hs + EY);
}

function harmony_experience_score(color1, color2) {
  let hcl1 = d3.hcl(color1);
  let hcl2 = d3.hcl(color2);

  let delta_C = Math.sqrt(
    Math.pow(hcl1.h - hcl2.h, 2) + Math.pow((hcl1.c - hcl2.c) / 1.46, 2)
  );
  let Hc = 0.04 + 0.53 * Math.tanh(0.8 - 0.045 * delta_C);

  let HLsum = 0.28 + 0.54 * Math.tanh(-3.88 + 0.029 * (hcl1.l + hcl2.l));
  let HdeltaL = 0.14 + 0.15 * Math.tanh(-2 + 0.2 * Math.abs(hcl1.l - hcl2.l));
  let HL = HLsum + HdeltaL;

  let HH = E_SY(hcl1.h, hcl1.c, hcl1.l) + E_SY(hcl2.h, hcl2.c, hcl2.l);

  return Hc + HL + HH;
}

function judge_harmony(palette) {
  let res = {};
  res["template_dist"] =
    (matsuda.check_templates(palette.map(d => d3.hsl(d).h))[0] * Math.PI) / 180;
  let pp_score = 0;
  let he_score = 0;
  let number_quad = 0;
  for (let i = 0; i < palette.length; i++) {
    for (let j = i + 1; j < palette.length; j++) {
      pp_score += getPreferenceScore(palette[i], palette[j]);
      he_score += harmony_experience_score(palette[i], palette[j]);
      number_quad++;
    }
  }
  res["number_quad"] = number_quad;
  if (number_quad === 0) {
    res["pp_score"] = 100000;
    res["he_score"] = 100000;
  } else {
    res["pp_score"] = pp_score / number_quad;
    res["he_score"] = he_score / number_quad;
  }
  res["number"] = palette.length;
  //   console.log("palette_harmony", palette, res);
  return res;
}

function summary_color_harmony(palettes) {
  let total_res = {};
  let total_num = 0;
  let bi_total_num = 0;
  let total_num_quad = 0;
  let total_pp_score = 0;
  let total_he_score = 0;
  let total_template_dist = 0;
  palettes.forEach(palette => {
    let res = judge_harmony(palette);
    total_num += res["number"];
    total_num_quad += res["number_quad"];
    if (res["number_quad"] > 0) {
      total_pp_score += res["pp_score"] * res["number"];
      total_he_score += res["he_score"] * res["number"];
      bi_total_num += res["number"];
    }
    total_template_dist += res["template_dist"] * res["number"];
  });
  if (bi_total_num === 0) {
    total_res["pp_score"] = 100000;
    total_res["he_score"] = 100000;
  } else {
    total_res["pp_score"] = total_pp_score / bi_total_num;
    total_res["he_score"] = total_he_score / bi_total_num;
  }
  total_res["template_dist"] = total_template_dist / total_num;
  total_res["number"] = total_num;
  total_res["number_quad"] = total_num_quad;
  return total_res;
}

function save_result(res) {
  // from local storage check whether the result is already there
  let result = localStorage.getItem("result");
  if (result === null) {
    result = [];
  } else {
    result = JSON.parse(result);
  }
  result.push(res);
  localStorage.setItem("result", JSON.stringify(result));
}

function static_result(last_num) {
  let result = localStorage.getItem("result");
  if (result === null) {
    result = [];
  } else {
    result = JSON.parse(result);
  }
  let cur = result.slice(-last_num);

  // get the average and std
  let avg = {};
  let std = {};
  Object.keys(cur[0]).forEach(key => {
    let data = cur.map(d => d[key]);
    avg[key] = d3.mean(data);
    std[key] = d3.deviation(data);
  });
  // console.log("Avg", avg);
  // console.log("Std", std);
  return [avg, std];
}

function static_result2(start, end) {
  let result = localStorage.getItem("result");
  if (result === null) {
    result = [];
  } else {
    result = JSON.parse(result);
  }
  let cur = result.slice(start, end);
  let is_single = false;
  if (start === end - 1) {
    is_single = true;
  }

  // get the average and std
  let avg = {};
  let std = {};
  Object.keys(cur[0]).forEach(key => {
    let data = cur.map(d => d[key]).filter(d => d < 10000);
    if (data.length < 1) {
      avg[key] = 100000;
      std[key] = 0;
    } else {
      avg[key] = d3.mean(data);
      std[key] = is_single ? 0 : d3.deviation(data);
      if (std[key] === undefined) {
        std[key] = 0;
      }
    }
  });
  // console.log("Avg", avg);
  // console.log("Std", std);
  if (std["pp_score"] === undefined) {
    // console.log(start, end, cur, avg, std);
  }
  return [avg, std];
}

function clear_result() {
  localStorage.removeItem("result");
}

function all_result() {
  let result = localStorage.getItem("result");
  if (result === null) {
    result = [];
  } else {
    result = JSON.parse(result);
  }
  return result;
}

export {
  judge_color_dist,
  summary_color_dist,
  judge_harmony,
  summary_color_harmony,
  save_result,
  static_result,
  static_result2,
  clear_result,
  all_result
};
