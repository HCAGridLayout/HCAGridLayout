/* eslint-disable */
import * as d3 from "d3";
import { GridLayout } from "./layout_grid";

const GridRender = function(parent) {
  let that = this;
  that.parent = parent;
  that.layout = new GridLayout(this);
  that.init = true;
  that.in_update = true;
  that.is_zoomout = false;
  that.render_image = false;
  that.image_border = 4;
  that.image_history = new Set();
  that.image_records = new Set();
  that.boundary_border = 100;
  that.mode = ["alpha", "isolate_label"]; // 'alpha', 'boundary', 'isolate_label';
  that.pboundary_stroke = "gray";
  that.pboundary_stroke_width = 1;
  that.one_partition = false;
  that.hide_opacity = 0.4;
//  that.min_image_size = 44;
//  that.min_image_size2 = 80;

  that.current_hover = -1;

  let dark_bias = 0;
  let dark_rate = 0.33;
  let light_bias = 255;
  let light_rate = 0;
  let bar_rate = 1;


  that.update_info_from_parent = function() {
    that.grid_group = that.parent.grid_group;
    that.render_image = that.parent.use_image;

    that.create_ani = that.parent.create_ani;
    that.update_ani = that.parent.update_ani;
    that.remove_ani = that.parent.remove_ani;
    that.fast_ani = 400;
    that.colorinter = d3.interpolateHcl;
    that.svg_width = that.parent.svg_width;
    that.svg_height = that.parent.svg_height;
    that.gridstack = that.parent.gridstack;

    that.min_image_size = that.parent.min_image_size;
    that.min_image_size2 = that.parent.min_image_size2;
  };

  that.update_info_from_parent();

  that.evaluate = function(grid_info, color_set = null) {
    return that.layout.evaluate_color(grid_info, color_set);
  };

  that.render = function(grid_info, color_set = null, two_stage = false) {
    that.parent.addOnLoadingFlag();
    that.parent.addGridLoadingFlag();
//    console.log("+1");
    if (two_stage && color_set !== null) {
      that.grids = that.layout.update_color(that.grids, color_set);
    } else {
      // update info
      that.update_info_from_parent();

      // update state
      let [meta, grids] = that.layout.update_layout(grid_info, color_set);
      that.meta = meta;
      that.grids = grids;
//      console.log("render_", that.meta.max_pid);
      if (two_stage) return [meta, grids];
    }
    that.grid_width = that.layout.cell - 2 * that.layout.stroke_width
    that.image_border = max(4, 0.1*that.grid_width);
    if(that.grid_width - 2*that.image_border >= that.min_image_size)that.image_width = that.grid_width - 2*that.image_border;
    else that.image_width = that.min_image_size2;
    if(that.grid_width-2*that.image_border<that.image_width*0.9)that.image_has_stroke=true;
    else that.image_has_stroke=false;

//    console.log("render", that.meta);
    if (that.meta.max_pid === 1) {
      that.one_partition = true;
      that.parent.allow_details = false;
      // that.one_partition = false;
    } else {
      that.one_partition = false;
      that.parent.allow_details = true;
      // that.one_partition = true;
    }

    // update view
    that.grid_group.attr(
      "transform",
      `translate(${that.meta.delta_x}, ${that.meta.delta_y})`
    );
    that.init_recall();
    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);
    that.e_paths = that.grid_group
      .selectAll(".partition_boundary")
      .data(that.meta.paths, d => d.name);

    that.create();
    that.update();
    that.remove();
    if (that.init) that.init = false;
//    console.log("render_image", that.render_image, that.grids.length);
    if (that.render_image) {
      let chosen_names = [];
      let chosen_ids = [];
      that.grids.forEach(grid => {
        if (grid.show_image) {
          chosen_names.push(grid.name);
          chosen_ids.push(grid.index);
        }
      });
      // console.log(chosen_names, chosen_ids);
      that.parent.fetchImages({names: chosen_names, ids: chosen_ids, batch: true});
    } else that.image_history = that.image_records;

    setTimeout(function() {
      that.parent.decOnLoadingFlag();
      that.parent.decGridLoadingFlag();
//      console.log("-1");
    }, (that.init ? that.create_ani : that.remove_ani + that.update_ani + that.create_ani) + 100);
  };

  that.filter_grids = function(filter_area) {
    let filter_grids = that.grids.filter(grid => {
      let center_x = that.meta.delta_x + grid.x + grid.width / 2;
      let center_y = that.meta.delta_y + grid.y + grid.height / 2;
      return (
        center_x >= filter_area.x1 &&
        center_x <= filter_area.x2 &&
        center_y >= filter_area.y1 &&
        center_y <= filter_area.y2
      );
    });
    return filter_grids.map(d => d.name);
  };

  that.render_images = function(images, chosen_ids) {
    that.image_records = new Set();
    if (chosen_ids === null) {
      that.grids.forEach(grid => {
        grid.img = `data:image/jpeg;base64,${images[grid.name]}`;
        that.image_records.add(grid.sample_id);
      });
    } else {
      chosen_ids.forEach((id, i) => {
        that.grids[that.layout.id_map[id]].img = `data:image/jpeg;base64,${images[i]}`;
        that.image_records.add(that.grids[that.layout.id_map[id]].sample_id);
      });
      if (images.length === 1 && that.current_hover === that.grids[that.layout.id_map[chosen_ids[0]]].name) {
        let meta_image = document.querySelector('#meta-image');
        let image_container = document.querySelector('#meta-image-container');
//        meta_image.style.width = 'auto';
//        meta_image.style.height = 'auto';
        meta_image.src = `data:image/jpeg;base64,${images[0]}`;
        let containerRatio = image_container.clientWidth / image_container.clientHeight;
        let imageRatio = meta_image.naturalWidth / meta_image.naturalHeight;
        if (containerRatio < imageRatio) {
            meta_image.style.width = '100%';
            meta_image.style.height = 'auto';
        } else {
            meta_image.style.width = 'auto';
            meta_image.style.height = '100%';
        }
      }
    }

    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);

    that.e_grids
      .select("rect.image_rect")
      .filter(d => that.image_records.has(d.sample_id))
      .attr("opacity", 0)
      .style("visibility", "hidden")
      .transition()
      .duration(that.update_ani)
      .attr("x", d => d.image_bias[0] + (d.width-that.image_width)/2)
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2)
      .attr("width", d => that.image_width)
      .attr("height", d => that.image_width)
      .filter(d => d.show_image)
      .attr("opacity", that.parent.show_images ? 1 : 0)
      .style("visibility", that.parent.show_images ? "visible" : "hidden")
      .attr("stroke", that.image_has_stroke ? "white" : null)
      .attr("stroke-width", that.image_has_stroke ? 0.05*that.image_width : null);

    that.e_grids
      .select("image")
      .filter(d => that.image_records.has(d.sample_id))
      .attr("opacity", 0)
      .style("visibility", "hidden")
      .transition()
      .duration(that.update_ani)
//      .attr("x", that.image_border)
//      .attr("y", that.image_border)
//      .attr("width", d => d.width - 2 * that.image_border)
//      .attr("height", d => d.height - 2 * that.image_border)
      .attr("x", d => d.image_bias[0] + (d.width-that.image_width)/2)
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2)
      .attr("width", d => that.image_width)
      .attr("height", d => that.image_width)
      .filter(d => d.show_image)
      .attr("opacity", that.parent.show_images ? 1 : 0)
      .style("visibility", that.parent.show_images ? "visible" : "hidden")
      .attr("xlink:href", d => d.img);

    that.e_grids
      .select("path.pin")
      .filter(d => that.image_records.has(d.sample_id))
      .attr("opacity", 0)
      .style("visibility", "hidden")
      .transition()
      .duration(that.update_ani)
        .attr("transform", d => `scale(${d.width/1024*2/3}) translate(${1024*(1/(2/3)-1)/2}, ${1024*(1/(2/3)-1)/2})`)
      .filter(d => d.show_image)
      .style("visibility", that.parent.show_images ? "visible" : "hidden")
      .attr("opacity", d => (that.parent.show_images && d.use_image_bias) ? 1 : 0);

    that.image_history = that.image_records;
  };

  that.init_recall = function() {
    that.grid_group
      .on("mousedown", ev => {
        if (!that.parent.in_overview) return;
        that.parent.overview_recall_mousedown(ev);
      })
      .on("mousemove", ev => {
        if (!that.parent.in_overview) return;
        that.parent.overview_recall_mousemove(ev);
      })
      .on("mouseup", ev => {
        if (!that.parent.in_overview) return;
        that.parent.overview_recall_mouseup(ev);
      });
  };

  that.create = function() {
    that.cur_pclass = null;
    // grid partition boundary
    that.e_paths
      .enter()
      .append("path")
      .attr("class", "partition_boundary")
      .attr("d", d => d.path)
      // .attr('stroke', d => `rgb(${d.pcolor[0]},${d.pcolor[1]},${d.pcolor[2]})`)
      .attr("stroke", that.pboundary_stroke)
      .attr("stroke-width", that.pboundary_stroke_width)
      .attr("opacity", 0)
      .attr("fill", "none");

    // grid group boundary
    let boundary_group = that.grid_group
      .selectAll(".grid_boundary")
      .data([that.meta]);
    boundary_group
      .enter()
      .append("g")
      .attr("class", "grid_boundary")
      .on("mouseenter", () => {
        if (that.in_update || that.one_partition || that.parent.show_details)
          return;
        if (that.cur_pclass !== null) {
          if (that.mode.indexOf("boundary") !== -1) {
            that.e_paths = that.grid_group
              .selectAll(".partition_boundary")
              .data(that.meta.paths, d => d.name);
            that.e_paths
              .filter(e => e.pclass === that.cur_pclass)
              .transition()
              .duration(that.fast_ani)
              .attr("opacity", 0);
          }

          that.e_grids = that.grid_group
            .selectAll(".grid-cell")
            .data(that.grids, d => d.sample_id);
          // console.log(that.cur_pclass, that.e_grids.filter(e => e.pclass === that.cur_pclass));
          that.e_grids
            // .filter(e => e.pclass === that.cur_pclass)
            .transition()
            .duration(that.fast_ani)
            .attr("transform", d => `translate(${d.x}, ${d.y})`);
          that.e_grids
            // .filter(e => e.pclass === that.cur_pclass)
            .select("rect.main")
            .transition()
            .duration(that.fast_ani)
            .attr("fill-opacity", 1)
            .attrTween("fill", that.colorinter)
            .attr(
              "fill",
              e => `rgba(${e.pcolor[0]},${e.pcolor[1]},${e.pcolor[2]})`
            )
            .attr("width", d => d.width)
            .attr("height", d => d.height);

          that.e_grids
            .selectAll("rect.confuse-1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
//            .attr("stroke", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`);
//            .attr("stroke", "white");

          that.e_grids
            .selectAll("rect.confuse0")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
//            .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
            .attr("fill", 'none')
            .attr("stroke", e => `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})`);

          that.e_grids
            .selectAll("rect.confuse1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})`);

          that.cur_pclass = null;
        }
      })
      .append("rect")
      .attr("x", d => d.minx - that.boundary_border)
      .attr("y", d => d.miny - that.boundary_border)
      .attr("width", d => d.maxx - d.minx + 2 * that.boundary_border)
      .attr("height", d => d.maxy - d.miny + 2 * that.boundary_border)
      .attr("fill", "none")
      .attr("stroke", "gray")
      .attr("stroke-width", that.boundary_border * 2 - 0.5)
      .attr("opacity", 0);
    // .transition()
    // .duration(that.create_ani)
    // .attr('opacity', 1);

    function extractString(str) {
      const regex = /\+(.*?)\./;
      const match = str.match(regex);
      let s = (match ? match[1] : str).replace(/_/g, ' ');
      let i, ss = s.toLowerCase().split(/\s+/);
      for (i = 0; i < ss.length; i++) {
        ss[i] = ss[i].slice(0, 1).toUpperCase() + ss[i].slice(1);
      }
      return ss.join(' ');
    }
    // grid layouts
    let e_grid_groups = that.e_grids
      .enter()
      .append("g")
      .attr("class", "grid-cell")
      .attr("transform", d =>
        that.one_partition || that.is_zoomout
          ? `translate(${d.px[d.pclass]}, ${d.py[d.pclass]})`
          : `translate(${d.x}, ${d.y})`
      )
      .on("mouseover", function(ev, d) {
        // d3.select(this).raise()
        //     .select('rect')
        //     .attr('stroke', 'black')
        //     .attr('stroke-width', d => d.stroke_width * 2);
        // console.log('over', d);

        //d3.select("#meta1").text(`${d.label}`);
        let ss = `${extractString(d.label_name)}`;
        if(d.label!=d.bottom_label) {
          ss = ss + ` (${extractString(d.bottom_label_name)})`
        }
        d3.select("#meta1t").text(`${extractString(d.label_name)}`)
          .attr("title", ss);
        //d3.select("#meta2").text(`${d.gt_label}`);
        let gt_ss = `${extractString(d.gt_label_name)}`;
        if(d.gt_label!=d.bottom_gt_label) {
          gt_ss = gt_ss + ` (${extractString(d.bottom_gt_label_name)})`
        }
        d3.select("#meta2t").text(`${extractString(d.gt_label_name)}`)
          .attr("title", gt_ss);
//        if (d.is_confused) {
//          let tmp_s = 'yes:';
////          for(let i=0;i<min(3, d.confuse_labels.length);i++)tmp_s = tmp_s + `  ${d.confuse_labels[i]}-${d.confuse_label_names[i]}-${d.confuse_values[i]},`
////          d3.select("#meta3").text(
////            `yes: ${d.confuse_label} ${d.confuse_label_name} - ${d.confuse_value}`
////          );
//          d3.select("#meta3").text(`Yes`);
//        } else {
//          d3.select("#meta3").text(`No`);
//        }
        that.current_hover = d.name;
        if (d.img === '') {
          that.parent.fetchImages({names: [d.name], ids: [d.index]});
        }
        else {
          let meta_image = document.querySelector('#meta-image');
          let image_container = document.querySelector('#meta-image-container');
          meta_image.src = d.img;
          let containerRatio = image_container.clientWidth / image_container.clientHeight;
          let imageRatio = meta_image.naturalWidth / meta_image.naturalHeight;
          if (containerRatio < imageRatio) {
              meta_image.style.width = '100%';
              meta_image.style.height = 'auto';
          } else {
              meta_image.style.width = 'auto';
              meta_image.style.height = '100%';
          }
        }

        let meta_bar = d3.select("#meta-bar");

        let svgElement = document.getElementById('meta-bar');
        let svgWidth = svgElement.clientWidth;
        let svgHeight = svgElement.clientHeight;
        svgElement.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);

        meta_bar = meta_bar.select("g");

        meta_bar.select(".bar-boundary")
          .attr("width", svgWidth)
          .attr("height", svgHeight);

        let bar_data = [];
        let full_conf = 0;
        let label_keys = Object.keys(that.layout.colors);
        label_keys.forEach(function(label) {
          let find = -1;
          for(let i=0;i<d.confuse_labels.length;i++) {
            if(d.confuse_labels[i]==label){
              find = i;
              break;
            }
          }
          if(find>=0){
            bar_data.push({"label_id": label, "conf": d.confuse_values[find], "bias_conf": full_conf, "color": that.layout.colors[label]});
            full_conf += d.confuse_values[find];
          }
        });
        bar_data.push({"label_id": -1, "conf": max(0, 1-full_conf), "bias_conf": full_conf, "color": [127, 127, 127]});

        let update = meta_bar.selectAll(".bar")
            .data(bar_data, d => d.label_id);

        update.enter().append("rect")
          .attr("class", "bar")
          .attr("x", d => 0.05*svgHeight + (svgWidth-0.1*svgHeight)*d.bias_conf)
          .attr("width", d => (svgWidth-0.1*svgHeight)*d.conf)
          .attr("y", d => 0.05*svgHeight)
          .attr("height", d => 0.9*svgHeight)
          .attr("fill", d => `rgba(${d.color[0]},${d.color[1]},${d.color[2]})`);

        update.exit().remove();

        update.transition()
          .duration(300)
          .attr("x", d => 0.05*svgHeight + (svgWidth-0.1*svgHeight)*d.bias_conf)
          .attr("width", d => (svgWidth-0.1*svgHeight)*d.conf)
          .attr("y", d => 0.05*svgHeight)
          .attr("height", d => 0.9*svgHeight)
          .attr("fill", d => `rgba(${d.color[0]},${d.color[1]},${d.color[2]})`);

        if (that.in_update || that.one_partition || that.parent.show_details)
          return;
        if (d.pclass !== that.cur_pclass) {
          // console.log('over', d);
          if (that.mode.indexOf("boundary") !== -1) {
            that.e_paths = that.grid_group
              .selectAll(".partition_boundary")
              .data(that.meta.paths, d => d.name);
            that.e_paths
              .filter(e => e.pclass === d.pclass)
              .raise()
              .transition()
              .duration(that.fast_ani)
              .attr("opacity", 1);
            that.e_paths
              .filter(e => e.pclass === that.cur_pclass)
              .transition()
              .duration(that.fast_ani)
              .attr("opacity", 0);
          }

          that.e_grids = that.grid_group
            .selectAll(".grid-cell")
            .data(that.grids, d => d.sample_id);
          // if (that.cur_pclass !== null) {
          //     that.e_grids
          //         .filter(e => e.pclass === that.cur_pclass)
          //         .select('rect')
          //         .transition()
          //         .duration(that.fast_ani)
          //         .attrTween('fill', that.colorinter)
          //         .attr('fill', e => `rgb(${e.pcolor[0] * 255},${e.pcolor[1] * 255},${e.pcolor[2] * 255}), 0.5`);
          // }
          that.e_grids
            .transition()
            .duration(that.fast_ani)
            .attr(
              "transform",
              e => `translate(${e.px[d.pclass]}, ${e.py[d.pclass]})`
            );

          that.e_grids
            .filter(e => e.pclass === d.pclass)
            .select("rect.main")
            .transition()
            .duration(that.fast_ani)
            .attr("fill-opacity", 1)
            .attrTween("fill", that.colorinter)
            .attr(
              "fill",
              e => `rgba(${e.color[0]},${e.color[1]},${e.color[2]})`
            )
            .attr("width", e => e.pwidth[d.pclass])
            .attr("height", e => e.pheight[d.pclass]);

          if (that.mode.indexOf("alpha") !== -1) {
            that.e_grids
              .filter(e => e.pclass !== d.pclass)
              .select("rect.main")
              .transition()
              .duration(that.fast_ani)
              .attr("fill-opacity", that.hide_opacity)
              .attrTween("fill", that.colorinter)
              .attr(
                "fill",
                e => `rgba(${e.pcolor[0]},${e.pcolor[1]},${e.pcolor[2]})`
              )
              .attr("width", e => e.pwidth[d.pclass])
              .attr("height", e => e.pheight[d.pclass]);
          } else {
            that.e_grids
              .filter(e => e.pclass !== d.pclass)
              .select("rect.main")
              .transition()
              .duration(that.fast_ani)
              .attrTween("fill", that.colorinter)
              .attr(
                "fill",
                e => `rgba(${e.pcolor[0]},${e.pcolor[1]},${e.pcolor[2]})`
              )
              .attr("width", e => e.pwidth[d.pclass])
              .attr("height", e => e.pwidth[d.pclass]);
          }

          that.e_grids
            .filter(e => e.pclass === d.pclass)
            .selectAll("rect.confuse-1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//            .attr("stroke", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`);
//            .attr("stroke", "white");

          that.e_grids
            .filter(e => e.pclass === d.pclass)
            .selectAll("rect.confuse0")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
//            .attr("fill", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
            .attr("fill", 'none')
            .attr("stroke", e => `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`);

          that.e_grids
            .filter(e => e.pclass === d.pclass)
            .selectAll("rect.confuse1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

          that.e_grids
            .filter(e => e.pclass !== d.pclass)
            .selectAll("rect.confuse-1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
//            .attr("stroke", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`);
//            .attr("stroke", "white");

          that.e_grids
            .filter(e => e.pclass !== d.pclass)
            .selectAll("rect.confuse0")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
//            .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
            .attr("fill", 'none')
            .attr("stroke", e => `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})`);

          that.e_grids
            .filter(e => e.pclass !== d.pclass)
            .selectAll("rect.confuse1")
            .transition()
            .duration(that.fast_ani)
            .attrTween("fill", that.colorinter)
            .attr("fill", e => `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})`);

          that.cur_pclass = d.pclass;
        }
      })
//      .on("click", function(ev, d) {
//        // console.log("click", d);
//        if (that.in_update || that.one_partition || that.parent.show_details)
//          return;
//        that.in_update = true;
//        let sample_ids = that.grids
//          .filter(e => e.pclass === d.pclass)
//          .map(e => e.name);
//        console.log("zoom in", sample_ids);
//        let args = {
//          samples: sample_ids,
//          zoom_without_expand: false,
//        };
//        that.parent.fetchGridLayout(args);
//      })
      .style("cursor", "pointer")
      .attr("opacity", 0);

    e_grid_groups
      .transition()
      .delay(that.init ? 0 : that.remove_ani + that.update_ani)
      .duration(that.create_ani)
      .attr("opacity", 1)
      .on("end", () => {
        that.in_update = false;
      });

//    console.log("ani time", that.create_ani);

    if (!that.one_partition && !that.is_zoomout) {
      e_grid_groups
        .append("rect")
        .attr("class", "main")
        .attr("width", d => d.width)
        .attr("height", d => d.height)
        .attr("fill-opacity", 1)
        .attr("fill", d => `rgba(${d.pcolor[0]},${d.pcolor[1]},${d.pcolor[2]})`)
        .attr("stroke", "white")
        .attr("stroke-width", d => d.stroke_width);
    } else {
      e_grid_groups
        .append("rect")
        .attr("class", "main")
        .attr("fill-opacity", 1)
        .attr("fill", e => `rgba(${e.color[0]},${e.color[1]},${e.color[2]})`)
        .attr("width", e => e.pwidth[e.pclass])
        .attr("height", e => e.pheight[e.pclass])
        .attr("stroke", "white")
        .attr("stroke-width", d => d.stroke_width);
    }

    let e_grid_bar = e_grid_groups
      .append("g")
      .attr("class", "confuse-bar grid-bar");

    e_grid_bar
      .style("visibility", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? "visible" : "hidden")
      .attr("opacity", 0)
      .transition()
      .delay(that.init ? 0 : that.remove_ani + that.update_ani)
      .duration(that.create_ani)
      .attr("opacity", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? 1 : 0);

    e_grid_bar
      .append("rect")
      .attr("class", "confuse-1")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
//      .attr("width", d => d.width * 0.9)
//      .attr("height", d => d.width * 0.9)
//      .attr("x", d => d.width * 0.05)
//      .attr("y", d => d.width * 0.05)
//      .attr("rx", d => d.width * 0.05)
//      .attr("ry", d => d.width * 0.05)
      .attr("width", d => d.width)
      .attr("height", d => d.width)
      .attr("x", 0)
      .attr("y", 0)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", "white")
//      .attr("stroke-width", d => d.width * 0.08);

    e_grid_bar
      .append("rect")
      .attr("class", "confuse1")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => d.width * 0.9)
      .attr("height", d => (d.confuse_values[0]) * (d.width * 0.9))
      .attr("x", d => d.width * 0.05)
      .attr("y", d => d.width * 0.05 + d.width*0.9*(1-d.confuse_values[0]))
      .attr("rx", d => d.width * 0.05)
      .attr("ry", d => d.width * 0.05)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})` : `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

    e_grid_bar
      .append("rect")
      .attr("class", "confuse0")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => d.width * 0.9)
      .attr("height", d => d.width * 0.9)
      .attr("x", d => d.width * 0.05)
      .attr("y", d => d.width * 0.05)
      .attr("rx", d => d.width * 0.05)
      .attr("ry", d => d.width * 0.05)
      .attr("fill-opacity", 1)
      .attr("fill", 'none')
      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})` : `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`)
      .attr("stroke-width", d => d.width * 0.03);

    if (that.render_image) {
      e_grid_groups
        .append("rect")
        .attr("class", "image_rect")
        .attr("fill", "none")
        .attr("x", d => d.image_bias[0] + (d.width-that.image_width)/2)
        .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2)
        .attr("width", d => that.image_width)
        .attr("height", d => that.image_width)
        .attr("opacity", 0)
        .style("visibility", "hidden");
      e_grid_groups
//        .append("image")
//        .attr("x", that.image_border)
//        .attr("y", that.image_border)
//        .attr("width", d => d.width - 2 * that.image_border)
//        .attr("height", d => d.height - 2 * that.image_border);
        .append("image")
        .attr("x", d => d.image_bias[0] + (d.width-that.image_width)/2)
        .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2)
        .attr("width", d => that.image_width)
        .attr("height", d => that.image_width)
        .attr("preserveAspectRatio", "none")
        .attr("opacity", 0)
        .style("visibility", "hidden");
      e_grid_groups
        .append("path")
        .attr("class", "pin")
        .attr("d", "M512 64.383234c-189.077206 0-342.355289 153.643944-342.355289 343.172854S512 959.616766 512 959.616766s342.355289-362.531768 342.355289-552.060679S701.077206 64.383234 512 64.383234zM512 497.079441c-65.76594 0-119.080367-53.44115-119.080367-119.364471S446.23406 258.350499 512 258.350499s119.080367 53.44115 119.080367 119.364471S577.76594 497.079441 512 497.079441z")
        .attr("fill", "#2c2c2c")
        .attr("stroke", "white")
        .attr("stroke-width", 100)
        .attr("transform", d => `scale(${d.width/1024*2/3}) translate(${1024*(1/(2/3)-1)/2}, ${1024*(1/(2/3)-1)/2})`)
        .attr("opacity", 0)
        .style("visibility", "hidden");
    }

    let e_image_bar = e_grid_groups
      .append("g")
      .attr("class", "confuse-bar image-bar");

    e_image_bar
      .style("visibility", d => (d.is_confused && d.show_image && that.parent.show_images) ? "visible" : "hidden")
      .attr("opacity", 0)
      .transition()
      .delay(that.init ? 0 : that.remove_ani + that.update_ani)
      .duration(that.create_ani)
      .attr("opacity", d => (d.is_confused && d.show_image && that.parent.show_images) ? 1 : 0);

    e_image_bar
      .append("rect")
      .attr("class", "confuse-1")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => that.image_width*0.6)
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.05*that.image_width)
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
      .attr("stroke", "white")
      .attr("stroke-width", d => that.image_width*0.08);

    e_image_bar
      .append("rect")
      .attr("class", "confuse1")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
//      .attr("width", d => that.image_width*0.17)
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => (d.confuse_values[0]) * (that.image_width*0.57))
//      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.055*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.17+0.055*that.image_width))
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.035*that.image_width + that.image_width*0.57*(1-d.confuse_values[0]))
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})` : `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

    e_image_bar
      .append("rect")
      .attr("class", "confuse0")
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => that.image_width*0.6)
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.05*that.image_width)
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
      .attr("fill-opacity", 1)
//      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
      .attr("fill", 'none')
      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})` : `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`)
      .attr("stroke-width", d => that.image_width*0.03);
  };

  that.update = function() {
    if (!that.one_partition && !that.is_zoomout) {
      that.e_grids
        .transition()
        .ease(d3.easeSin)
        .duration(that.update_ani)
        .delay(that.init ? 0 : that.remove_ani)
        .attr("transform", d => `translate(${d.x}, ${d.y})`);
      that.e_grids
        .select("rect.main")
        .transition()
        .duration(that.update_ani)
        .delay(that.init ? 0 : that.remove_ani)
        .attr("width", d => d.width)
        .attr("height", d => d.height)
        .attrTween("fill", that.colorinter)
        .attr("fill", d => `rgba(${d.pcolor[0]},${d.pcolor[1]},${d.pcolor[2]})`)
        .on("end", () => {
          that.in_update = false;
        });
    } else {
      that.e_grids
        .transition()
        .ease(d3.easeSin)
        .duration(that.update_ani)
        .delay(that.init ? 0 : that.remove_ani)
        .attr(
          "transform",
          d => `translate(${d.px[d.pclass]}, ${d.py[d.pclass]})`
        );
      that.e_grids
        .select("rect.main")
        .transition()
        .duration(that.update_ani)
        .delay(that.init ? 0 : that.remove_ani)
        .attr("width", e => e.pwidth[e.pclass])
        .attr("height", e => e.pheight[e.pclass])
        .attrTween("fill", that.colorinter)
        .attr("fill", d => `rgba(${d.color[0]},${d.color[1]},${d.color[2]})`)
        .on("end", () => {
          that.in_update = false;
          if (that.is_zoomout) that.is_zoomout = false;
        });
    }

    let e_grid_bar = that.e_grids
      .select(".grid-bar");

    if(!that.init) {
      e_grid_bar
        .transition()
        .duration(that.remove_ani)
        .attr("opacity", 0)
        .on("end", function(d) {
          d3.select(this)
            .style("visibility", (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? "visible" : "hidden")
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani)
            .attr("opacity", (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? 1 : 0)
            .attr("attr", (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)));
        });
    }else {
      e_grid_bar
        .style("visibility", "visible")
        .transition()
        .duration(that.create_ani)
        .delay(that.init ? 0 : that.remove_ani+that.update_ani)
        .attr("opacity", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? 1 : 0)
        .on("end", function(d) {
          d3.select(this)
            .style("visibility", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? "visible" : "hidden");
        });
    }

    e_grid_bar
      .select("rect.confuse-1")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
//      .attr("width", d => d.width * 0.9)
//      .attr("height", d => d.width * 0.9)
//      .attr("x", d => d.width * 0.05)
//      .attr("y", d => d.width * 0.05)
//      .attr("rx", d => d.width * 0.05)
//      .attr("ry", d => d.width * 0.05)
      .attr("width", d => d.width)
      .attr("height", d => d.width)
      .attr("x", 0)
      .attr("y", 0)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", "white")
//      .attr("stroke-width", d => d.width * 0.08);

    e_grid_bar
      .select("rect.confuse0")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => d.width * 0.9)
      .attr("height", d => d.width * 0.9)
      .attr("x", d => d.width * 0.05)
      .attr("y", d => d.width * 0.05)
      .attr("rx", d => d.width * 0.05)
      .attr("ry", d => d.width * 0.05)
      .attr("fill", 'none')
      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})` : `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`)
      .attr("stroke-width", d => d.width * 0.03);

    e_grid_bar
      .select("rect.confuse1")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => d.width * 0.9)
      .attr("height", d => (d.confuse_values[0]) * (d.width * 0.9))
      .attr("x", d => d.width * 0.05)
      .attr("y", d => d.width * 0.05 + d.width*0.9*(1-d.confuse_values[0]))
      .attr("rx", d => d.width * 0.05)
      .attr("ry", d => d.width * 0.05)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})` : `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

    let e_image_bar = that.e_grids
      .select(".image-bar");

    if(!that.init) {
      e_image_bar
        .transition()
        .duration(that.remove_ani)
        .attr("opacity", 0)
        .on("end", function(d) {
          d3.select(this)
            .style("visibility", (d.is_confused && d.show_image && that.parent.show_images) ? "visible" : "hidden")
            .transition()
            .duration(that.create_ani)
            .delay(that.update_ani)
            .attr("opacity", (d.is_confused && d.show_image && that.parent.show_images) ? 1 : 0);
        });
    }else {
      e_image_bar
        .style("visibility", "visible")
        .transition()
        .duration(that.create_ani)
        .delay(that.init ? 0 : that.remove_ani+that.update_ani)
        .attr("opacity", d => (d.is_confused && d.show_image && that.parent.show_images) ? 1 : 0)
        .on("end", function(d) {
          d3.select(this)
            .style("visibility", (d.is_confused && d.show_image && that.parent.show_images) ? "visible" : "hidden");
        });
    }

    e_image_bar
      .select("rect.confuse-1")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => that.image_width*0.6)
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.05*that.image_width)
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
      .attr("stroke", "white")
      .attr("stroke-width", d => that.image_width*0.08);

    e_image_bar
      .select("rect.confuse0")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => that.image_width*0.6)
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.05*that.image_width)
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
//      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})` : `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
      .attr("fill", 'none')
      .attr("stroke", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})` : `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`)
      .attr("stroke-width", d => that.image_width*0.03);

    e_image_bar
      .select("rect.confuse1")
      .transition()
      .duration(that.update_ani)
      .delay(that.init ? 0 : that.remove_ani)
      .attr("opacity", d => (d.is_confused ? 1 : 0))
//      .attr("width", d => that.image_width*0.17)
      .attr("width", d => that.image_width*0.2)
      .attr("height", d => (d.confuse_values[0]) * (that.image_width*0.57))
//      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.055*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.17+0.055*that.image_width))
      .attr("x", d => d.image_bias[0] + (d.image_bias[0]<=0 ? min(0, (d.width-that.image_width)/2)-0.07*that.image_width : d.width - min(0, (d.width-that.image_width)/2)-that.image_width*0.2+0.07*that.image_width))
      .attr("y", d => d.image_bias[1] + (d.width-that.image_width)/2-0.035*that.image_width + that.image_width*0.57*(1-d.confuse_values[0]))
      .attr("rx", d => that.image_width*0.05)
      .attr("ry", d => that.image_width*0.05)
      .attr("fill-opacity", 1)
      .attr("fill", e => (!that.one_partition && !that.is_zoomout) ? `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})` : `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

    that.e_grids
      .select("image")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .style("visibility", "hidden");

    that.e_grids
      .select("rect.image_rect")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .style("visibility", "hidden");

    that.e_grids
      .select("path.pin")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .style("visibility", "hidden");

    if (!(that.render_image)) {
      //      that.e_grids
      //        .select("image")
      ////        .filter(d => !that.image_history.has(d.sample_id))
      //        .transition()
      //        .duration(that.update_ani)
      //        .attr("opacity", 1);
      ////        .transition()
      //        .duration(that.update_ani)
      //        .delay(that.init ? 0 : that.remove_ani)
      //        .attr("href", d => d.img)
      //        .attr("opacity", 1);
      //    } else {
      that.e_grids
        .select("rect.image_rect")
        .attr("opacity", 0)
        .style("visibility", "hidden");
      that.e_grids
        .select("image")
        //        .transition()
        //        .duration(that.remove_ani)
        .attr("opacity", 0)
        .style("visibility", "hidden");
      //        .attr("href", "");
      that.e_grids
        .select("path.pin")
        .attr("opacity", 0)
        .style("visibility", "hidden");
    }

    let selection = that.grid_group
      .selectAll(".grid-cell")
      .filter(d => d.show_image);

    selection = selection.sort(function(a, b) {
      return a.order - b.order;
    });

    selection
      .each(function(d) {
//        console.log(d.order, d.x, d.y);
        d3.select(this).raise();
      });
  };

  that.remove = function() {
    that.e_paths
      .exit()
      .transition()
      .duration(that.remove_ani)
      .attr("opacity", 0)
      .remove();

    that.e_grids
      .exit()
      .transition()
      .duration(that.remove_ani)
      .attr("opacity", 0)
      .remove();
  };

  that.show_images = function() {
//    console.log("show", that.parent.show_images);

    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);

    that.e_grids
      .select("rect.image_rect")
      .filter(d => d.show_image)
      .style("visibility", "visible")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 1);

    that.e_grids
      .select("image")
      .filter(d => d.show_image)
      .style("visibility", "visible")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 1);

    that.e_grids
      .select("path.pin")
      .filter(d => d.show_image)
      .style("visibility", "visible")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", d => d.use_image_bias ? 1 : 0);

    that.e_grids
      .select(".grid-bar")
      .filter(d => d.show_image)
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? 1 : 0)
      .on("end", function() {
        d3.select(this).style("visibility", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? "visible" : "hidden");
      });

    that.e_grids
      .select(".image-bar")
      .filter(d => d.show_image)
      .style("visibility", d => (d.is_confused && d.show_image && that.parent.show_images) ? "visible" : "hidden")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", d => (d.is_confused && d.show_image && that.parent.show_images) ? 1 : 0);
  }

  that.hide_images = function() {
//    console.log("hide", that.parent.show_images);

    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);

    that.e_grids
      .select("rect.image_rect")
      .filter(d => d.show_image)
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .on("end", function() {
        d3.select(this).style("visibility", "hidden");
      });

    that.e_grids
      .select("image")
      .filter(d => d.show_image)
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .on("end", function() {
        d3.select(this).style("visibility", "hidden");
      });

    that.e_grids
      .select("path.pin")
      .filter(d => d.show_image)
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", 0)
      .on("end", function() {
        d3.select(this).style("visibility", "hidden");
      });

    that.e_grids
      .select(".grid-bar")
      .filter(d => d.show_image)
      .style("visibility", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? "visible" : "hidden")
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", d => (d.is_confused && (!that.parent.show_images || !d.show_image || d.use_image_bias)) ? 1 : 0);

    that.e_grids
      .select(".image-bar")
      .filter(d => d.show_image)
      .transition()
      .duration(that.fast_ani)
      .attr("opacity", d => (d.is_confused && d.show_image && that.parent.show_images) ? 1 : 0)
      .on("end", function() {
        d3.select(this).style("visibility", d => (d.is_confused && d.show_image && that.parent.show_images) ? "visible" : "hidden");
      });
  }

  that.show_details = function() {
    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);

    that.e_grids
      .select("rect.main")
      .transition()
      .duration(that.update_ani)
      .attr("fill-opacity", 1)
      .attr("fill", e => `rgba(${e.color[0]},${e.color[1]},${e.color[2]})`)
//      .attr("width", e => e.pwidth[e.pclass])
//      .attr("height", e => e.pheight[e.pclass])
      .attr("stroke", "white")
      .attr("stroke-width", d => d.stroke_width);

    that.e_grids
      .selectAll("rect.confuse-1")
      .transition()
      .duration(that.update_ani)
      .attr("fill", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
//      .attr("stroke", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`);
//      .attr("stroke", "white");

    that.e_grids
      .selectAll("rect.confuse0")
      .transition()
      .duration(that.update_ani)
//      .attr("fill", e => `rgba(${light_bias+e.color[0]*light_rate},${light_bias+e.color[1]*light_rate},${light_bias+e.color[2]*light_rate})`)
      .attr("fill", 'none')
      .attr("stroke", e => `rgba(${dark_bias+e.color[0]*dark_rate},${dark_bias+e.color[1]*dark_rate},${dark_bias+e.color[2]*dark_rate})`);

    that.e_grids
      .selectAll("rect.confuse1")
      .transition()
      .duration(that.update_ani)
      .attr("fill", e => `rgba(${e.color[0]*bar_rate},${e.color[1]*bar_rate},${e.color[2]*bar_rate})`);

//    that.e_grids
//      .transition()
//      .duration(that.update_ani)
//      .attr(
//        "transform",
//        d => `translate(${d.px[d.pclass]}, ${d.py[d.pclass]})`
//      );
  };

  that.hide_details = function() {
    that.e_grids = that.grid_group
      .selectAll(".grid-cell")
      .data(that.grids, d => d.sample_id);

    that.e_grids
      .select("rect.main")
      .transition()
      .duration(that.update_ani)
      .attr("width", d => d.width)
      .attr("height", d => d.height)
      .attr("fill-opacity", 1)
      .attr("fill", d => `rgba(${d.pcolor[0]},${d.pcolor[1]},${d.pcolor[2]})`)
      .attr("stroke", "white")
      .attr("stroke-width", d => d.stroke_width);

    that.e_grids
      .selectAll("rect.confuse-1")
      .transition()
      .duration(that.update_ani)
      .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
//      .attr("stroke", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`);
//      .attr("stroke", "white");

    that.e_grids
      .selectAll("rect.confuse0")
      .transition()
      .duration(that.update_ani)
//      .attr("fill", e => `rgba(${light_bias+e.pcolor[0]*light_rate},${light_bias+e.pcolor[1]*light_rate},${light_bias+e.pcolor[2]*light_rate})`)
      .attr("fill", 'none')
      .attr("stroke", e => `rgba(${dark_bias+e.pcolor[0]*dark_rate},${dark_bias+e.pcolor[1]*dark_rate},${dark_bias+e.pcolor[2]*dark_rate})`);

    that.e_grids
      .selectAll("rect.confuse1")
      .transition()
      .duration(that.update_ani)
      .attr("fill", e => `rgba(${e.pcolor[0]*bar_rate},${e.pcolor[1]*bar_rate},${e.pcolor[2]*bar_rate})`);

    that.e_grids
      .transition()
      .duration(that.update_ani)
      .attr("transform", d => `translate(${d.x}, ${d.y})`);
  };
};

export default GridRender;
