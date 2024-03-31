/*eslint-disable*/
// import {simulatedAnnealing2FindBestPalette} from './optimizeFunc';
// import {extend_in_same_hue} from './hue_extension';
import findBestPaletteBetwClasses from "./optimizeFunc_betwclass";
import findBestPaletteGlobal from "./optimizeFunc_global";
import { generateByPaletailor } from "./palettailor";
import {
  judge_color_dist,
  summary_color_dist,
  judge_harmony,
  summary_color_harmony,
  save_result
} from "./evaluate";
import * as d3 from "d3";

const ColorGenerater = function(parent) {
  let that = this;
  that.parent = parent;
  that.base_colors = [
    [248, 182, 187], // pink
    [171, 227, 246], // blue
    [216, 193, 246], // purple
    [243, 228, 191], // yellow
    [185, 243, 190], // green
    [252, 245, 155],
    [221, 221, 221],
    [138, 170, 208],
    [191, 185, 134],
    [255, 193, 152],
    [127, 137, 253],
    [255, 136, 104],
    [175, 203, 191],
    [170, 167, 188],
    [254, 228, 179]
  ];
  that.gray_color = [221, 221, 221];
  that.cur_base_id = 0;
  that.color_sets = {};
  that.colorsscope = {
    hue_scope: [0, 360],
    chroma_scope: [20, 100],
    lumi_scope: [15, 85]
  };

  that.is_init = true;
  that.generate_palette = async function(
    color_ids,
    cur_node_id = 0,
    parent_node_id = -1,
    type = "simulated_annealing",
    regenerate = true,
    resave = false,
    driven_data = null,
    filter_ratio = 0.005,
    use_palettailor = true
  ) {
    if (parent_node_id === -1 && !regenerate) {
      await that.parent.fetchColorMap();
    }
    // console.log('load color', that.parent.colorstack.length, that.parent.gridstack.length);

    let delta = that.parent.colorstack.length - that.parent.gridstack.length;
    while (delta > 0) {
      that.parent.popColormap();
      delta -= 1;
    }
    if (delta === 0) {
      // console.log("load color successfully");
      return that.parent.colorstack[that.parent.colorstack.length - 1];
    }

    // get time
    // let start_time = new Date().getTime();
    let color_set = {};
    let partition_color_set = {};
    let parent_color_set = {};
    if (parent_node_id > -1) {
      let parent_colormaps =
        that.parent.colorstack[that.parent.colorstack.length - 1];
      parent_color_set = parent_colormaps["colorset"];
    }
    let labels = Object.keys(color_ids);
    let label_nums = that.get_labels_num(labels, driven_data);
    let filter_num = Math.round(label_nums["total"] * filter_ratio);
    // console.log('palette, labels', labels, color_ids);
    let id2label = {};
    labels.forEach(label => {
      id2label[color_ids[label]] = Number(label);
    });
    let type_labels = [[], {}, [], []];
    let parent_set = new Set();
    labels.forEach(label => {
      let color_id = color_ids[label];
      if (color_id in parent_color_set) {
        type_labels[0].push(label);
        return;
      }
      if (label_nums[label] < filter_num) {
        type_labels[3].push(label);
        return;
      }
      let last_index = color_id.lastIndexOf("-");
      if (last_index === -1) {
        type_labels[2].push(label);
        return;
      }
      let parent_id = color_id.substring(0, last_index);
      let child_id = color_id.substring(last_index + 1);
      if (!(parent_id in parent_color_set)) {
        // if (parent_node_id > -1) {
        //     // console.log('palette node', parent_id, child_id, parent_color_set);
        //     console.error('parent_id not in color_set');
        //     return;
        // }
        parent_color_set[parent_id] = that.base_colors[that.cur_base_id];
        that.cur_base_id = (that.cur_base_id + 1) % that.base_colors.length;
      }
      parent_set.add(parent_id);
      type_labels[1][parent_id] = type_labels[1][parent_id] || [];
      type_labels[1][parent_id].push(child_id);
    });

    // colors in previous layer
    type_labels[0].forEach(label => {
      color_set[color_ids[label]] = parent_color_set[color_ids[label]];
      partition_color_set[color_ids[label]] = color_set[color_ids[label]];
    });

    // colors needed to be extended
    let l1_objs = Object.keys(type_labels[1]);
    l1_objs.sort(
      (d, e) => -type_labels[1][d].length + type_labels[1][e].length
    );
    // first modify parent colors
    let parent_colors_info = l1_objs.map((parent_id, idx) => {
      let children = type_labels[1][parent_id];
      children.sort();
      return {
        id: parent_id,
        color: parent_color_set[parent_id],
        rgb: d3.rgb(...parent_color_set[parent_id]),
        num: type_labels[1][parent_id].length,
        children: children,
        child_labels: children.map(id => id2label[parent_id + "-" + id]),
        index: idx
      };
    });

    if (!use_palettailor) {
      // console.log('start parent');
      for (let i = 0; i < 1; i++) {
        let start_time = new Date().getTime();
        let end_time = 0,
          use_time = 0;
        if (parent_colors_info.length > 0) {
          const bestParentColors = findBestPaletteBetwClasses(
            parent_colors_info.map(info => info["rgb"]),
            that.colorsscope,
            driven_data,
            parent_colors_info.map(info => info["child_labels"])
          );
          end_time = new Date().getTime();
          use_time = end_time - start_time;
          bestParentColors.id.forEach((color, index) => {
            parent_colors_info[index]["rgb"] = color;
            parent_colors_info[index]["color"] = [color.r, color.g, color.b];
            parent_color_set[parent_colors_info[index]["id"]] =
              parent_colors_info[index]["color"];
          });
        }

        parent_colors_info.forEach((info, index) => {
          let exclude_colors = [];
          parent_colors_info.forEach((info2, index2) => {
            if (index === index2) {
              return;
            }
            exclude_colors.push(info2["rgb"]);
          });
          info["excludes"] = exclude_colors;
        });

        // global optimization
        let start_time2 = new Date().getTime();
        const bestGlobalColors = findBestPaletteGlobal(
          parent_colors_info.map(info => info["rgb"]),
          parent_colors_info.map(info => info["num"]),
          null,
          driven_data,
          parent_colors_info.map(info => info["child_labels"])
        ).id;
        let end_time2 = new Date().getTime();
        let use_time2 = end_time2 - start_time2;
        // console.log("generate palette time:", use_time + use_time2);
        let start_idx = 0;
        let all_palettes = [];
        parent_colors_info.forEach(info => {
          let parent_id = info["id"];
          let part_palette = bestGlobalColors.slice(
            start_idx,
            start_idx + info["num"]
          );
          all_palettes.push(part_palette);
          part_palette.forEach((color, index) => {
            color_set[parent_id + "-" + info["children"][index]] = [
              color.r,
              color.g,
              color.b
            ];
            partition_color_set[
              parent_id + "-" + info["children"][index]
            ] = info["color"].slice();
          });
          start_idx += info["num"];
        });
        // console.log(
        //   "all_palettes",
        //   all_palettes,
        //   summary_color_dist(all_palettes),
        //   summary_color_harmony(all_palettes)
        // );
        let total_Colors = [];
        all_palettes.forEach(palette => {
          palette.forEach(color => {
            total_Colors.push(color);
          });
        });
        // console.log(
        //   "total_Colors",
        //   total_Colors,
        //   judge_color_dist(total_Colors),
        //   judge_harmony(total_Colors)
        // );
        let summary_res = {};
        // let dist_res = judge_color_dist(total_Colors);
        // let harmony_res = judge_harmony(total_Colors);
        let dist_res = summary_color_dist(all_palettes);
        let harmony_res = summary_color_harmony(all_palettes);
        Object.keys(dist_res).forEach(key => {
          summary_res[key] = dist_res[key];
        });
        Object.keys(harmony_res).forEach(key => {
          summary_res[key] = harmony_res[key];
        });
        summary_res["time"] = use_time + use_time2;
        let total_dist = judge_color_dist(total_Colors);
        summary_res["total_min_dist"] = total_dist.min_perception_difference;
        summary_res["total_name_dist"] = total_dist.min_name_difference;
        // console.log("global_palette", total_Colors, summary_res);
        save_result(summary_res);
      }
    } else {
      // console.log('Extand in same hue', driven_data);
      // parent_colors_info.forEach(info => {
      //     let parent_id = info['id'];
      //     const bestColors = extend_in_same_hue(info['rgb'], info['num']);
      //     bestColors.forEach((color, index) => {
      //         color_set[parent_id + '-' + info['children'][index]] = [color.r, color.g, color.b];
      //         partition_color_set[parent_id + '-' + info['children'][index]] = info['color'].slice();
      //     });
      // });
      // console.log(d3.version);
      console.log("Palettailor", driven_data);
      // get avg dist of parent colors
      // let avg_dist = 0;
      // let parent_colors = parent_colors_info.map(info => d3.hcl(info['rgb']));
      // for (let i = 0; i < parent_colors.length; i++) {
      //     for (let j = i + 1; j < parent_colors.length; j++) {
      //         avg_dist += d3_ciede2000(parent_colors[i], parent_colors[j]);
      //     }
      // }
      // avg_dist /= parent_colors.length * (parent_colors.length - 1) / 2;
      for (let i = 0; i < 1; i++) {
        let all_palettes = [];
        // get time
        let start_time = new Date().getTime();
        parent_colors_info.forEach(info => {
          let parent_id = info["id"];
          const bestColorsMap = generateByPaletailor(
            info["rgb"],
            driven_data.grids.filter(d =>
              info["child_labels"].includes(d.label)
            ),
            5 * (i + 1)
          );
          const bestColors = info["child_labels"].map(
            label => bestColorsMap[label]
          );
          all_palettes.push(bestColors);
          // console.log("bestColors", bestColors, judge_color_dist(bestColors));
          bestColors.forEach((color, index) => {
            color_set[parent_id + "-" + info["children"][index]] = [
              color.r,
              color.g,
              color.b
            ];
            partition_color_set[
              parent_id + "-" + info["children"][index]
            ] = info["color"].slice();
          });
        });
        let end_time = new Date().getTime();
        let use_time = end_time - start_time;
        console.log("generate palette time:", use_time);
        // console.log(
        //   "all_palettes",
        //   all_palettes,
        //   summary_color_dist(all_palettes),
        //   summary_color_harmony(all_palettes)
        // );
        let total_Colors = [];
        all_palettes.forEach(palette => {
          palette.forEach(color => {
            total_Colors.push(color);
          });
        });
        // console.log(
        //   "total_Colors",
        //   total_Colors,
        //   judge_color_dist(total_Colors),
        //   judge_harmony(total_Colors)
        // );
        let summary_res = {};
        // let dist_res = judge_color_dist(total_Colors);
        // let harmony_res = judge_harmony(total_Colors);
        let dist_res = summary_color_dist(all_palettes);
        let harmony_res = summary_color_harmony(all_palettes);
        Object.keys(dist_res).forEach(key => {
          summary_res[key] = dist_res[key];
        });
        Object.keys(harmony_res).forEach(key => {
          summary_res[key] = harmony_res[key];
        });
        summary_res["time"] = use_time;
        let total_dist = judge_color_dist(total_Colors);
        summary_res["total_min_dist"] = total_dist.min_perception_difference;
        summary_res["total_name_dist"] = total_dist.min_name_difference;
        console.log("global_palette", total_Colors, summary_res);
        save_result(summary_res);
      }
    }

    // other new colors
    type_labels[2].forEach(label => {
      color_set[color_ids[label]] = that.base_colors[that.cur_base_id];
      that.cur_base_id = (that.cur_base_id + 1) % that.base_colors.length;
      partition_color_set[color_ids[label]] = color_set[color_ids[label]];
    });

    // colors with few grids
    type_labels[3].forEach(label => {
      color_set[color_ids[label]] = that.gray_color;
      let last_index = color_ids[label].lastIndexOf("-");
      if (last_index > -1) {
        let parent_id = color_ids[label].substring(0, last_index);
        if (parent_set.has(parent_id)) {
          partition_color_set[color_ids[label]] = parent_color_set[parent_id];
          return;
        }
      }
      partition_color_set[color_ids[label]] = that.gray_color;
    });

    let colormaps = {};
    let colormap = {};
    let partitionmap = {};
    labels.forEach(label => {
      let color = color_set[color_ids[label]];
      colormap[label] = [color[0], color[1], color[2]];
      let pcolor = partition_color_set[color_ids[label]];
      partitionmap[label] = [pcolor[0], pcolor[1], pcolor[2]];
    });
    // let end_time = new Date().getTime();
    // console.log("generate palette time:", end_time - start_time);
    colormaps["colorset"] = color_set;
    colormaps["colormap"] = colormap;
    colormaps["partitionmap"] = partitionmap;

    that.parent.pushColormap(colormaps);
    if (parent_node_id === -1 && (!regenerate || resave)) {
      that.parent.fetchColorMap(colormaps);
    }
    return colormaps;
  };

  that.palettailor = function(
    color_ids,
    parent_color_set = {},
    driven_data = null,
    seed = 0
  ) {
    // get time
    // let start_time = new Date().getTime();
    let color_set = {};
    let partition_color_set = {};
    let labels = Object.keys(color_ids);
    let label_nums = that.get_labels_num(labels, driven_data);
    let filter_ratio = 0.0025;
    let filter_num = Math.round(label_nums["total"] * filter_ratio);
    // console.log('palette, labels', labels, color_ids);
    let id2label = {};
    labels.forEach(label => {
      id2label[color_ids[label]] = Number(label);
    });
    let type_labels = [[], {}, [], []];
    let parent_set = new Set();
    labels.forEach(label => {
      let color_id = color_ids[label];
      if (color_id in parent_color_set) {
        type_labels[0].push(label);
        return;
      }
      if (label_nums[label] < filter_num) {
        type_labels[3].push(label);
        return;
      }
      let last_index = color_id.lastIndexOf("-");
      if (last_index === -1) {
        type_labels[2].push(label);
        return;
      }
      let parent_id = color_id.substring(0, last_index);
      let child_id = color_id.substring(last_index + 1);
      if (!(parent_id in parent_color_set)) {
        // if (parent_node_id > -1) {
        //     // console.log('palette node', parent_id, child_id, parent_color_set);
        //     console.error('parent_id not in color_set');
        //     return;
        // }
        parent_color_set[parent_id] = that.base_colors[that.cur_base_id];
        that.cur_base_id = (that.cur_base_id + 1) % that.base_colors.length;
      }
      parent_set.add(parent_id);
      type_labels[1][parent_id] = type_labels[1][parent_id] || [];
      type_labels[1][parent_id].push(child_id);
    });

    // colors in previous layer
    type_labels[0].forEach(label => {
      color_set[color_ids[label]] = parent_color_set[color_ids[label]];
      partition_color_set[color_ids[label]] = color_set[color_ids[label]];
    });

    // colors needed to be extended
    let l1_objs = Object.keys(type_labels[1]);
    l1_objs.sort(
      (d, e) => -type_labels[1][d].length + type_labels[1][e].length
    );
    // first modify parent colors
    let parent_colors_info = l1_objs.map((parent_id, idx) => {
      let children = type_labels[1][parent_id];
      children.sort();
      return {
        id: parent_id,
        color: parent_color_set[parent_id],
        rgb: d3.rgb(...parent_color_set[parent_id]),
        num: type_labels[1][parent_id].length,
        children: children,
        child_labels: children.map(id => id2label[parent_id + "-" + id]),
        index: idx
      };
    });

    // get avg dist of parent colors
    // let avg_dist = 0;
    // let parent_colors = parent_colors_info.map(info => d3.hcl(info['rgb']));
    // for (let i = 0; i < parent_colors.length; i++) {
    //     for (let j = i + 1; j < parent_colors.length; j++) {
    //         avg_dist += d3_ciede2000(parent_colors[i], parent_colors[j]);
    //     }
    // }
    // avg_dist /= parent_colors.length * (parent_colors.length - 1) / 2;
    // for (let i = 0; i < 1; i++) {
    let all_palettes = [];
    // get time
    let start_time = new Date().getTime();
    parent_colors_info.forEach(info => {
      let parent_id = info["id"];
      const bestColorsMap = generateByPaletailor(
        info["rgb"],
        driven_data.grids.filter(d => info["child_labels"].includes(d.label)),
        seed
      );
      const bestColors = info["child_labels"].map(
        label => bestColorsMap[label]
      );
      all_palettes.push(bestColors);
      // console.log("bestColors", bestColors, judge_color_dist(bestColors));
      bestColors.forEach((color, index) => {
        color_set[parent_id + "-" + info["children"][index]] = [
          color.r,
          color.g,
          color.b
        ];
        partition_color_set[parent_id + "-" + info["children"][index]] = info[
          "color"
        ].slice();
      });
    });
    let end_time = new Date().getTime();
    let use_time = end_time - start_time;
    console.log("generate palette time:", use_time);

    // let total_Colors = [];
    // all_palettes.forEach(palette => {
    //   palette.forEach(color => {
    //     total_Colors.push(color);
    //   });
    // });
    // let summary_res = {};
    // // let dist_res = judge_color_dist(total_Colors);
    // // let harmony_res = judge_harmony(total_Colors);
    // let dist_res = summary_color_dist(all_palettes);
    // let harmony_res = summary_color_harmony(all_palettes);
    // Object.keys(dist_res).forEach(key => {
    //   summary_res[key] = dist_res[key];
    // });
    // Object.keys(harmony_res).forEach(key => {
    //   summary_res[key] = harmony_res[key];
    // });
    // summary_res["time"] = use_time;
    // let total_dist = judge_color_dist(total_Colors);
    // summary_res["total_min_dist"] = total_dist.min_perception_difference;
    // summary_res["total_name_dist"] = total_dist.min_name_difference;
    // console.log("global_palette", total_Colors, summary_res);
    // save_result(summary_res);
    // }

    // other new colors
    type_labels[2].forEach(label => {
      color_set[color_ids[label]] = that.base_colors[that.cur_base_id];
      that.cur_base_id = (that.cur_base_id + 1) % that.base_colors.length;
      partition_color_set[color_ids[label]] = color_set[color_ids[label]];
    });

    // colors with few grids
    type_labels[3].forEach(label => {
      color_set[color_ids[label]] = that.gray_color;
      let last_index = color_ids[label].lastIndexOf("-");
      if (last_index > -1) {
        let parent_id = color_ids[label].substring(0, last_index);
        if (parent_set.has(parent_id)) {
          partition_color_set[color_ids[label]] = parent_color_set[parent_id];
          return;
        }
      }
      partition_color_set[color_ids[label]] = that.gray_color;
    });

    let colormaps = {};
    let colormap = {};
    let partitionmap = {};
    labels.forEach(label => {
      let color = color_set[color_ids[label]];
      colormap[label] = [color[0], color[1], color[2]];
      let pcolor = partition_color_set[color_ids[label]];
      partitionmap[label] = [pcolor[0], pcolor[1], pcolor[2]];
    });
    // let end_time = new Date().getTime();
    // console.log("generate palette time:", end_time - start_time);
    colormaps["colormap"] = colormap;
    colormaps["partitionmap"] = partitionmap;
    colormaps["time"] = use_time / 1000.0;
    return colormaps;
  };

  that.get_labels_num = function(labels, data) {
    // console.log(labels, data);
    let labels_num = {};
    let total_num = 0;
    labels.forEach(label => {
      let labeln = Number(label);
      labels_num[label] = data["grids"].filter(d => d.label === labeln).length;
      total_num += labels_num[label];
    });
    labels_num["total"] = total_num;
    // console.log(labels_num);
    return labels_num;
  };
};

export default ColorGenerater;
