/* eslint-disable */
import * as d3 from "d3";
import {
  judge_color_dist,
  judge_harmony,
  summary_color_dist,
  summary_color_harmony,
  save_result
} from "./evaluate";

const GridLayout = function(parent) {
  let that = this;
  that.parent = parent;

  that.grid_margin = 0;
  that.stroke_width = 0.4; // 0.4;
  that.extra_width = 1.2; // 1.2
  that.partition_width = 3.0; // 3.0

  that.update_info_from_parent = function() {
    that.create_ani = that.parent.create_ani;
    that.update_ani = that.parent.update_ani;
    that.remove_ani = that.parent.remove_ani;
    that.layout_width = that.parent.svg_width - that.grid_margin * 2;
    that.layout_height = that.parent.svg_height - that.grid_margin * 2;
    that.gridstack = that.parent.gridstack;
    that.mode = that.parent.mode;
  };

  that.update_color = function(grids, color_set) {
    let colors = color_set["colormap"];
    let pcolors = color_set["partitionmap"];
    that.colors = colors;
    that.pcolors = pcolors;
    grids.forEach(grid => {
      grid.color = colors[grid.label];
      grid.pcolor = pcolors[grid.label];
    });
    return grids;
  };

  that.evaluate_color = function(grid_info, color_set) {
    let colors = color_set ? color_set["colormap"] : grid_info.colors;
    let pcolors = color_set ? color_set["partitionmap"] : grid_info.pcolors;
    that.colors = colors;
    that.pcolors = pcolors;
    let palettes = {};
    let all_colors = {};
    Object.keys(colors).forEach(key => {
      let pvalue = `${pcolors[key][0]}, ${pcolors[key][1]}, ${pcolors[key][2]}`;
      if (palettes[pvalue] === undefined) palettes[pvalue] = [];
      let value = `${colors[key][0]}, ${colors[key][1]}, ${colors[key][2]}`;
      if (all_colors[value] === undefined) {
        palettes[pvalue].push(d3.rgb(...colors[key]));
        all_colors[value] = 1;
      }
    });
    let palette_list = [];
    let palette_all = [];
    Object.keys(palettes).forEach(key => {
      palette_list.push(palettes[key]);
      palette_all = palette_all.concat(palettes[key]);
    });

    let summary_res = {};
    // let dist_res = judge_color_dist(palette_all);
    // let harmony_res = judge_harmony(palette_all);
    let dist_res;
    let harmony_res;
    dist_res = summary_color_dist(palette_list);
    harmony_res = summary_color_harmony(palette_list);
    Object.keys(dist_res).forEach(key => {
      summary_res[key] = dist_res[key];
    });
    Object.keys(harmony_res).forEach(key => {
      summary_res[key] = harmony_res[key];
    });
    let total_dist = judge_color_dist(palette_all);
    summary_res["total_min_dist"] = total_dist.min_perception_difference;
    summary_res["total_name_dist"] = total_dist.min_name_difference;
    if (color_set && color_set["time"]) {
      summary_res["time"] = color_set["time"];
    }
//    console.log("global_palette", palette_all, summary_res);
    save_result(summary_res);
  };

  that.update_layout = function(grid_info, color_set = null) {
    that.update_info_from_parent();

    // total position
    that.size = grid_info.size;
    that.cell = Math.min(
      that.layout_width / that.size[0],
      (that.layout_height-(1/2*that.parent.min_image_size2)) / that.size[1]
    );
    that.grid_width = that.cell * that.size[0];
    that.grid_height = that.cell * that.size[1];
    that.grid_margin_x =
      (that.layout_width - that.grid_width) / 2 + that.grid_margin;
    that.grid_margin_y =
      (that.layout_height - that.grid_height) / 2 + that.grid_margin;
    let meta = {
      delta_x: that.grid_margin_x,
      delta_y: that.grid_margin_y,
      grid_width: that.grid_width,
      grid_height: that.grid_height
    };

    // cell position and color
    let grids = [];
    console.assert(grid_info.grid.length === that.size[0] * that.size[1]);
    let x = 0;
    let y = 0;
    let colors = color_set ? color_set["colormap"] : {...grid_info.colors};
    let pcolors = color_set ? color_set["partitionmap"] : {...grid_info.pcolors};
    that.colors = colors;
    that.pcolors = pcolors;
    // console.log('color_set', color_set, colors, pcolors);
    if (color_set === null) {
      Object.keys(colors).forEach(key => {
        let raw = colors[key];
        colors[key] = [255 * raw[0], 255 * raw[1], 255 * raw[2]];
      });
      Object.keys(pcolors).forEach(key => {
        let raw = pcolors[key];
        pcolors[key] = [255 * raw[0], 255 * raw[1], 255 * raw[2]];
      });
    }
    let pclasses = {};
    let pid = 0;
    // console.log(color_set, grid_info.colors);
    class Matrix {
      constructor() {
        this.values = {};
      }
      set = function(x, y, value) {
        this.values[`${x},${y}`] = value;
      };
      get = function(x, y) {
        return this.values[`${x},${y}`];
      };
    }
    let matrix_t = new Matrix();
    let rconf_label_map = {};
    Object.keys(grid_info.confusion.conf_vis.labelmap).forEach(key => {
      rconf_label_map[grid_info.confusion.conf_vis.labelmap[key]] = key;
    });

    function argsort(arr) {
        // 创建一个索引数组
        var indices = new Array(arr.length);
        for (var i = 0; i < arr.length; i++) {
            indices[i] = i;
        }
        // 使用排序函数对索引数组进行排序，根据对应的元素值从大到小排序
        indices.sort(function(a, b) {
            return arr[b] - arr[a];
        });
        return indices;
    }

    let max_conf = [];
    for(let i=0;i<grid_info.confusion.conf_vis.confs.length;i++)max_conf.push(Math.max(...(grid_info.confusion.conf_vis.confs[i])));
    let conf_argsort = argsort(max_conf);
    //let sheld = min(0.8, max_conf[conf_argsort[Math.floor(max_conf.length*3/4)]]);
    let sheld = that.parent.parent.thresholdValue;
//    console.log("show thresholdValue", sheld);
    let if_confuse = [];
    for(let i=0;i<max_conf.length;i++)if_confuse.push(max_conf[i]<sheld ? true : false);

    let pos_bias_list = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]];
    let poses = [];
    let image_bias = [];
    let use_bias = false;
    let grid_width = that.cell - 2*that.stroke_width;
    let image_border = max(4, 0.1*grid_width);
    if((that.parent.min_image_size > grid_width-2*image_border)
      && (that.parent.min_image_size2 > that.cell*1.2))
      use_bias = true;
    for(let i=0;i<grid_info.grid.length;i++) {
      poses.push([0, 0]);
      image_bias.push([0, 0]);
    }
    for(let i=0;i<grid_info.grid.length;i++){
      let id = grid_info.grid[i];
      let lb1 = grid_info.labels[id];
      poses[id][0] = Math.floor(i / that.size[1]);
      poses[id][1] = i % that.size[1];
      if(use_bias) {
        let best_same = 0;
        let best_bias = 0;
        for(let j=0;j<8;j++){
          // if((j>=3)&&(j<=4))continue;
          let same_cnt = 0;
//          for(let k=-1;k<=1;k++) {
//            let kk = (j+8+k)%8;
//            let tmp_x = poses[id][0] + pos_bias_list[kk][0];
//            let tmp_y = poses[id][1] + pos_bias_list[kk][1];
//            if(tmp_x>=0 && tmp_x<that.size[0] && tmp_y>=0 && tmp_y<that.size[1]){
//              let lb2 = grid_info.labels[grid_info.grid[tmp_x*that.size[1]+tmp_y]];
//              if(lb1==lb2)same_cnt += 1;
//            }
//          }
          let kk = (j+8)%8;
          let range = [[0, 0], [0, 0]];
          for(let l=0;l<=1;l++){
            if(pos_bias_list[kk][l]==0)range[l] = [-Math.ceil(that.parent.min_image_size2/2/that.cell), Math.ceil(that.parent.min_image_size2/2/that.cell)];
            if(pos_bias_list[kk][l]==-1)range[l] = [-Math.ceil(that.parent.min_image_size2/that.cell), 0];
            if(pos_bias_list[kk][l]==1)range[l] = [0, Math.ceil(that.parent.min_image_size2/that.cell)];
          }
          for(let k=range[0][0];k<=range[0][1];k++)
          for(let l=range[1][0];l<=range[1][1];l++){
            let tmp_x = poses[id][0] + k;
            let tmp_y = poses[id][1] + l;
            if(tmp_x>=0 && tmp_x<that.size[0] && tmp_y>=0 && tmp_y<that.size[1]){
              let lb2 = grid_info.labels[grid_info.grid[tmp_x*that.size[1]+tmp_y]];
              if(lb1==lb2)same_cnt += 1;
            }else same_cnt -= 1;
          }
          if(same_cnt>best_same){
            best_same = same_cnt;
            best_bias = j;
          }
        }
        image_bias[id][0] = pos_bias_list[best_bias][0] * (that.parent.min_image_size2/2 - grid_width/4);
        image_bias[id][1] = pos_bias_list[best_bias][1] * (that.parent.min_image_size2/2 - grid_width/4);
      }
    }
    let image_width = 0;
    if(grid_width - 2*image_border >= that.parent.min_image_size)image_width = grid_width - 2*image_border;
    else image_width = that.parent.min_image_size2;
    let show_images = [];
    let if_show_images = [];
    for(let i=conf_argsort.length-1;i>=conf_argsort.length/2;i--){
      let id1 = conf_argsort[i];
      let x1 = poses[id1][0]*that.cell;
      let y1 = poses[id1][1]*that.cell;
      let b_x1 = image_bias[id1][0];
      let b_y1 = image_bias[id1][1];
      let flag = true;
      for(let j=0;j<show_images.length;j++){
        let id2 = show_images[j];
        let x2 = poses[id2][0]*that.cell;
        let y2 = poses[id2][1]*that.cell;
        let b_x2 = image_bias[id2][0];
        let b_y2 = image_bias[id2][1];
        if(max(Math.abs(x1+b_x1-x2-b_x2), Math.abs(y1+b_y1-y2-b_y2))*0.85<that.parent.min_image_size2){
          flag = false;
          break;
        }
      }
      if(flag){
        show_images.push(id1);
      }
      if(show_images.length*that.parent.min_image_size2*that.parent.min_image_size2>conf_argsort.length*that.cell*that.cell*0.25)break;
    }
//    console.log("show images num", show_images.length)
//    console.log("show images", show_images);
    for(let i=0;i<conf_argsort.length;i++)if_show_images.push(that.parent.min_image_size <= grid_width-2*image_border);
    for(let i=0;i<show_images.length;i++)if_show_images[show_images[i]] = true;

    that.id_map = [];
    let id_cnt = 0;
    grid_info.grid.forEach((d, i) => {
      let grid = {};
      grid.order = i;
      grid.pos = [x, y];
      grid.pos_t = [y, x];
      grid.x = y * that.cell + that.stroke_width;
      grid.y = x * that.cell + that.stroke_width;
      grid.width = that.cell - 2 * that.stroke_width;
      grid.height = that.cell - 2 * that.stroke_width;
//      grid.show_image = ((that.parent.render_image && grid_info.grid.length <= 400 && (x%3===0) && (y%3===0)) ? true : false);
      grid.show_image = if_show_images[d];
      grid.image_bias = [image_bias[d][1], image_bias[d][0]];
      grid.use_image_bias = use_bias;
      grid.name = d;
      grid.index = i;
      grid.label = grid_info.labels[d];
      grid.color_id = grid_info.color_ids[grid.label];
      if (grid.label === undefined) {
//        console.log("null grid");
        that.id_map.push(-1);
        y = (y + 1) % that.size[1];
        if (y === 0) x++;
        return;
      }
      that.id_map.push(id_cnt);
      id_cnt += 1;
      grid.label_name = grid_info.label_names[grid.label];
      grid.color = colors[grid.label];
      grid.pcolor = pcolors[grid.label];
      grid.gt_label = grid_info.gt_labels[d];
      grid.gt_label_name = grid_info.label_names[grid.gt_label];
      grid.bottom_label = grid_info.bottom_labels.labels[d];
      grid.bottom_label_name = grid_info.label_names[grid.bottom_label];
      grid.bottom_gt_label = grid_info.bottom_labels.gt_labels[d];
      grid.bottom_gt_label_name = grid_info.label_names[grid.bottom_gt_label];
      grid.is_confused = if_confuse[d];
//      let confuse_classes = grid_info.confusion.confuse_class[d];
//      let confuse_label = grid_info.confusion.conf_vis.labelmap[grid.gt_label];
      let confs = grid_info.confusion.conf_vis.confs[d];
//      grid.confuse_class =
//        confuse_classes[0] == confuse_label
//          ? confuse_classes[1]
//          : confuse_classes[0];
      grid.confuse_classes = argsort(confs);
//      grid.confuse_value =
//        confuse_classes[0] == confuse_label ? confs[1] : confs[0];
      grid.confuse_values = [];
      for(let i=0;i<confs.length;i++)grid.confuse_values.push(confs[grid.confuse_classes[i]]);
//      grid.confuse_label = rconf_label_map[grid.confuse_class];
      grid.confuse_labels = [];
      for(let i=0;i<confs.length;i++)grid.confuse_labels.push(rconf_label_map[grid.confuse_classes[i]]);
//      grid.confuse_label_name = grid_info.label_names[grid.confuse_label];
      grid.confuse_label_names = [];
      for(let i=0;i<confs.length;i++)grid.confuse_label_names.push(grid_info.label_names[grid.confuse_labels[i]]);
      grid.pstr =
        grid.color_id.indexOf("-") === -1
          ? grid.color_id
          : grid.color_id.substring(0, grid.color_id.lastIndexOf("-"));
      if (!(grid.pstr in pclasses)) {
        pclasses[grid.pstr] = [pid, grid.pcolor];
        pid++;
      }
      grid.pclass = pclasses[grid.pstr][0];
      grid.sample_id = grid_info.sample_ids[d];
      grid.stroke_width = that.stroke_width;
      grid.img = "";
      grids.push(grid);
      matrix_t.set(y, x, grid);
      y = (y + 1) % that.size[1];
      if (y === 0) x++;
    });
//    console.log(">>>", pclasses);
    grids.forEach(grid => {
      let pos_t = grid.pos_t;
      grid.left = matrix_t.get(pos_t[0] - 1, pos_t[1]);
      grid.right = matrix_t.get(pos_t[0] + 1, pos_t[1]);
      grid.top = matrix_t.get(pos_t[0], pos_t[1] - 1);
      grid.bottom = matrix_t.get(pos_t[0], pos_t[1] + 1);

      grid.px = [];
      grid.py = [];
      grid.pwidth = [];
      grid.pheight = [];

      // caculate extra width for label
      if (that.mode.indexOf("isolate_label") !== -1) {
        d3.range(pid).forEach(i => {
          let px = grid.x;
          let py = grid.y;
          let pwidth = grid.width;
          let pheight = grid.height;
          if (grid.pclass === i) {
            if (grid.left) {
              if (grid.left.pclass !== i) {
                px += that.partition_width;
                pwidth -= that.partition_width;
              } else if (grid.left.label !== grid.label) {
                px += that.extra_width;
                pwidth -= that.extra_width;
              }
            }
            if (grid.right) {
              if (grid.right.pclass !== i) pwidth -= that.partition_width;
              else if (grid.right.label !== grid.label)
                pwidth -= that.extra_width;
            }
            if (grid.top) {
              if (grid.top.pclass !== i) {
                py += that.partition_width;
                pheight -= that.partition_width;
              } else if (grid.top.label !== grid.label) {
                py += that.extra_width;
                pheight -= that.extra_width;
              }
            }
            if (grid.bottom) {
              if (grid.bottom.pclass !== i) pheight -= that.partition_width;
              else if (grid.bottom.label !== grid.label)
                pheight -= that.extra_width;
            }
          } else {
            if (grid.left && grid.left.pclass === i) {
              px += that.partition_width;
              pwidth -= that.partition_width;
            }
            if (grid.right && grid.right.pclass === i)
              pwidth -= that.partition_width;
            if (grid.top && grid.top.pclass === i) {
              py += that.partition_width;
              pheight -= that.partition_width;
            }
            if (grid.bottom && grid.bottom.pclass === i)
              pheight -= that.partition_width;
          }
          grid.px.push(px);
          grid.py.push(py);
          grid.pwidth.push(pwidth);
          grid.pheight.push(pheight);
        });
      }
    });

    let ppaths = [];
    let cur_node_index = that.gridstack[that.gridstack.length - 1];
    Object.keys(pclasses).forEach(pstr => {
      let pclass = pclasses[pstr][0];
      let boundaries = that.caculate_boundary(grids, pclass);
      let paths = that.caculate_path(boundaries, that.cell);
      paths.forEach((path, i) => {
        ppaths.push({
          pclass: pclass,
          path: path,
          name: `p-${cur_node_index}-${pclass}-${i}`,
          pcolor: pclasses[pstr][1]
        });
      });
    });
    meta.paths = ppaths;

    let minx, miny, maxx, maxy;
    grids.forEach((d, i) => {
      if (i === 0) {
        minx = d.x;
        miny = d.y;
        maxx = d.x + d.width;
        maxy = d.y + d.width;
      } else {
        minx = Math.min(minx, d.x);
        miny = Math.min(miny, d.y);
        maxx = Math.max(maxx, d.x + d.width);
        maxy = Math.max(maxy, d.y + d.width);
      }
    });
    meta.minx = minx;
    meta.miny = miny;
    meta.maxx = maxx;
    meta.maxy = maxy;
    meta.max_pid = pid;
    meta.matrix_t = matrix_t;

    // solve similarity
    let similarity_info = grid_info.similarity;
    meta.label_dists = similarity_info.matrix;
    meta.label_map = {};
    similarity_info.labels.forEach((d, i) => {
      meta.label_map[d] = i;
    });
    return [meta, grids];
  };

  that._get_set_first_value = function(set) {
    let [first] = set;
    return first;
  };

  that._get_edge_name = function(start_node, end_node) {
    let [x1, y1] = start_node.split(",");
    let [x2, y2] = end_node.split(",");
    if (x1 === x2) {
      if (parseInt(y1) < parseInt(y2)) return `td-${x1}-${y1}`;
      else return `td-${x2}-${y2}`;
    } else {
      if (parseInt(x1) < parseInt(x2)) return `lr-${x1}-${y1}`;
      else return `lr-${x2}-${y2}`;
    }
  };

  that.caculate_boundary = function(grids, pclass) {
    let pgrids = grids.filter(d => d.pclass === pclass);
    let edge_sets = new Set();
    // console.log(pgrids);
    pgrids.forEach(d => {
      let [y, x] = d.pos;
      let edges = [
        `td-${x}-${y}`,
        `td-${x + 1}-${y}`,
        `lr-${x}-${y}`,
        `lr-${x}-${y + 1}`
      ];
      edges.forEach(edge => {
        if (edge_sets.has(edge)) edge_sets.delete(edge);
        else edge_sets.add(edge);
      });
    });
    let edge_nodes = {};
    let node_links = {};
    edge_sets.forEach(edge => {
      // get edge start node and end node
      let [type, x, y] = edge.split("-");
      x = parseInt(x);
      y = parseInt(y);
      let start_node = `${x},${y}`;
      let end_node;
      if (type === "td") end_node = `${x},${y + 1}`;
      else end_node = `${x + 1},${y}`;
      edge_nodes[edge] = [start_node, end_node];
      // get node links
      if (!(start_node in node_links)) node_links[start_node] = new Set();
      if (!(end_node in node_links)) node_links[end_node] = new Set();
      node_links[start_node].add(end_node);
      node_links[end_node].add(start_node);
    });
    let boundaries = [];
    while (edge_sets.size > 0) {
      let cur_edge = that._get_set_first_value(edge_sets);
      edge_sets.delete(cur_edge);
      let boundary = [];
      let [start_node, end_node] = edge_nodes[cur_edge];
      boundary.push(start_node);
      boundary.push(end_node);
      let bf_node = start_node;
      let cur_node = end_node;
      while (cur_node !== start_node) {
        let cur_node_link = node_links[cur_node];
        cur_node_link.delete(bf_node);
        let next_node = that._get_set_first_value(cur_node_link);
        if (next_node === undefined) next_node = start_node;
        else cur_node_link.delete(next_node);
        cur_edge = that._get_edge_name(cur_node, next_node);
        edge_sets.delete(cur_edge);
        bf_node = cur_node;
        cur_node = next_node;
        boundary.push(cur_node);
      }
      boundaries.push(boundary);
    }
    // console.log('boundary', boundaries);
    return boundaries;
  };

  that.caculate_path = function(boundaries, cell) {
    let paths = boundaries.map(boundary => {
      let path = "";
      boundary.forEach((node, i) => {
        let [x, y] = node.split(",");
        x = parseInt(x);
        y = parseInt(y);
        if (i === 0) path += `M${x * cell},${y * cell}`;
        else path += `L${x * cell},${y * cell}`;
      });
      return path + "Z";
    });
    return paths;
  };
};

export { GridLayout };
