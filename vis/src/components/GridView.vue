<template>
  <div class='view' id='grid-container'>
      <div id='loading-background-grid' class="my-loading" style="display: block;"></div>
      <svg id='loading-svg-grid' version="1.1" width="40px" height="40px" style="enable-background:new 0 0 512 512; display: block;" xml:space="preserve"
        class="my-loading-svg">
        <symbol id="static-update-icon" viewBox="-150 -150 812 812">
          <g>
            <g>
              <path
                d="M463.702,162.655L442.491,14.164c-1.744-12.174-16.707-17.233-25.459-8.481l-30.894,30.894
                                          C346.411,12.612,301.309,0,254.932,0C115.464,0,3.491,109.16,0.005,248.511c-0.19,7.617,5.347,14.15,12.876,15.234l59.941,8.569
                                          c8.936,1.304,17.249-5.712,17.125-15.058C88.704,165.286,162.986,90,254.932,90c22.265,0,44.267,4.526,64.6,13.183l-29.78,29.78
                                          c-8.697,8.697-3.761,23.706,8.481,25.459l148.491,21.211C456.508,181.108,465.105,172.599,463.702,162.655z" />
            </g>
          </g>
          <g>
            <g>
              <path d="M499.117,249.412l-59.897-8.555c-7.738-0.98-17.124,5.651-17.124,16.143c0,90.981-74.019,165-165,165
                                          c-22.148,0-44.048-4.482-64.306-13.052l28.828-28.828c8.697-8.697,3.761-23.706-8.481-25.459L64.646,333.435
                                          c-9.753-1.393-18.39,6.971-16.978,16.978l21.21,148.492c1.746,12.187,16.696,17.212,25.459,8.481l31.641-31.626
                                          C165.514,499.505,210.587,512,257.096,512c138.794,0,250.752-108.618,254.897-247.28
                                          C512.213,257.088,506.676,250.496,499.117,249.412z" />
            </g>
          </g>
        </symbol>
        <symbol id="animate-update-icon" viewBox="0 0 60 60">
          <g transform="translate(30,30)">
            <path class="circle-path"
              d="M1.2246467991473533e-15,-20A20,20,0,1,1,-20,2.4492935982947065e-15L-16,1.959434878635765e-15A16,16,0,1,0,9.797174393178826e-16,-16Z">
            </path>
          </g>
        </symbol>
        <use xlink:href="#animate-update-icon" x="0" y="0" width="40px" height="40px"></use>
      </svg>
    <div id='main-gridlayout' class='gridlayout'>

      <svg class='grid-svg'>
        <g class='grid-group'>
          <!-- <rect class='grid-rect' v-for='i in 100' :key='i' :x='i*10' :y='i*10' width='10' height='10' fill='red' /> -->
        </g>
        <g class='overview-group'></g>
        <g class='confirm-group'></g>
      </svg>
      <div id='buttons'>
        <div id='cropping' title='Zoom in' @click='onCrop1Click' v-ripple class='button'>
          <svg class='icon' width='30px' height='30px' transform='translate(5, 5)' viewBox='0 0 1024 1024'>
            <path d="M477.663 848v64h-177v-64h177z m-235 0v64h-39.858C153.759 912 114 872.24 114 823.195v-48.044h64v48.044c0 13.7 11.105 24.805 24.805 24.805h39.858zM178 717.15h-64v-177h64v177z m0-235h-64v-177h64v177z m0-235h-64v-43.345C114 154.759 153.76 115 202.805 115h44.556v64h-44.556c-13.7 0-24.805 11.105-24.805 24.805v43.346zM305.36 179v-64h177v64h-177z m235 0v-64h177v64h-177z m235 0v-64h46.835C871.241 115 911 154.76 911 203.805v41.068h-64v-41.068c0-13.7-11.105-24.805-24.805-24.805h-46.834zM847 302.873h64v174.962h-64V302.873z m-57.059 439.485l112.271 112.271c12.497 12.497 12.497 32.758 0 45.255l-2.828 2.828c-12.497 12.497-32.758 12.497-45.255 0l-113-113L692.01 912.3 571.095 573.595 909.8 694.51l-119.858 47.848z" fill="white" p-id="1479"></path>
          </svg>
        </div>
        <div class='gap' v-show="!in_overview"></div>
        <div id='cropping2' title='Zoom in without expansion' @click='onCrop2Click' v-ripple class='button' v-show="!in_overview">
          <svg class='icon' width='30px' height='30px' transform='translate(5, 5)' viewBox='0 0 200 200'>
            <path d="M93.3,165.6v12.5H47.4v-12.5H93.3z M47.4,165.6v12.5h-7.8c-9.6,0-17.3-7.8-17.3-17.3v-9.4h12.5v9.4c0,2.7,2.2,4.8,4.8,4.8H47.4z M34.8,151.4H22.3v-45.9h12.5V151.4z M34.8,105.5H22.3V48.3l12.5,0V105.5z M34.8,48.3H22.3v-8.5c0-9.6,7.8-17.3,17.3-17.3h8.7V35h-8.7c-2.7,0-4.8,2.2-4.8,4.8V48.3L34.8,48.3z M48.3,35V22.5h57.2V35H48.3z M105.5,35V22.5h45.9l0,12.5H105.5z M151.4,35V22.5h9.1c9.6,0,17.3,7.8,17.3,17.3v8h-12.5v-8c0-2.7-2.2-4.8-4.8-4.8H151.4L151.4,35z M165.4,47.8h12.5v45.5h-12.5V47.8z M154.3,145l21.9,21.9c2.4,2.4,2.4,6.4,0,8.8l-0.6,0.6c-2.4,2.4-6.4,2.4-8.8,0l-22.1-22.1l-9.6,23.9L111.5,112l66.2,23.6L154.3,145L154.3,145z" fill="white" p-id="1479"></path>
          </svg>
        </div>
        <div class='gap' v-show="allow_zoomout"></div>
        <div id='zoomout' title='Zoom out' @click='onZoomOutClick' v-ripple v-show="allow_zoomout" class='button'>
          <svg t="1685968804315" class="icon" viewBox="0 0 1024 1024" width='30px' height='30px'
            transform='translate(5, 5)'>
            <path
              d="M312.533333 320l109.226667-109.226667-60.373333-60.373333L149.333333 362.666667l211.2 211.2 61.226667-61.866667-106.666667-106.666667A512 512 0 0 1 789.333333 874.666667h85.333334a597.333333 597.333333 0 0 0-562.133334-554.666667z"
              p-id="13973" fill="#ffffff"></path>
          </svg>
        </div>
        <div class='gap' v-show="allow_details"></div>
        <div id='details' title='Show/hide details' @click='onDetailsClick' v-ripple v-show="allow_details"
          class='button'>
          <svg t="1685968804315" class="icon" viewBox="0 0 1024 1024" width='30px' height='30px'
            transform='translate(5, 5)'>
            <path
              d="M131.84 698.221714h189.44v186.002286c0 20.992 12.434286 33.846857 32.585143 33.846857 19.712 0 32.128-12.854857 32.128-33.865143V698.221714H635.428571v186.002286c0 20.992 12.434286 33.846857 32.566858 33.846857 19.712 0 32.146286-12.854857 32.146285-33.865143V698.221714h192.420572c21.010286 0 33.865143-12.434286 33.865143-32.585143 0-20.132571-12.854857-32.146286-33.865143-32.146285H700.16V392.228571h192.420571c21.010286 0 33.865143-12.434286 33.865143-32.146285 0-20.132571-12.854857-32.548571-33.865143-32.548572H700.16V140.196571c0-21.010286-12.434286-34.285714-32.146286-34.285714-20.132571 0-32.566857 13.275429-32.566857 34.285714V327.497143H386.011429V140.214857c0-21.010286-12.434286-34.285714-32.146286-34.285714-20.150857 0-32.585143 13.275429-32.585143 34.285714V327.497143H131.84c-21.412571 0-34.267429 12.434286-34.267429 32.566857 0 19.730286 12.854857 32.146286 34.285715 32.146286h189.44v241.28H131.84c-21.430857 0-34.285714 12.013714-34.285714 32.146285 0 20.150857 12.854857 32.585143 34.285714 32.585143z m254.171429-64.731428V392.228571h249.417142v241.28z"
              p-id="13973" fill="#ffffff"></path>
          </svg>
        </div>
        <div class='gap' v-show="use_image"></div>
        <div id='images' title='Show/hide images' @click='onImageButtonClick' v-ripple v-show="use_image"
          class='button'>
          <svg t="1685968804315" class="icon" viewBox="0 0 1024 1024" width='30px' height='30px'
            transform='translate(5, 5)'>
            <path
              d="M831.792397 82.404802 191.548594 82.404802c-60.676941 0-110.042255 49.364291-110.042255 110.042255l0 640.245849c0 60.677964 49.364291 110.042255 110.042255 110.042255l640.244826 0c60.677964 0 110.042255-49.364291 110.042255-110.042255L941.835675 192.447057C941.834652 131.769093 892.470361 82.404802 831.792397 82.404802zM191.548594 122.420167l640.244826 0c38.612413 0 70.02689 31.414477 70.02689 70.02689l0 134.349871c-144.759965 4.953825-280.06151 63.59234-382.864898 166.396751-48.28061 48.28061-86.814228 103.732549-114.628714 163.962306-80.588433-68.744687-197.638289-73.051783-282.803971-12.938684L121.522728 192.447057C121.521704 153.834644 152.935158 122.420167 191.548594 122.420167zM121.521704 832.691883l0-136.601144c74.040297-72.025407 192.529945-71.925123 266.451538 0.301875-23.496134 62.998823-35.762505 130.383536-35.762505 199.672622 0 2.336208 0.420579 4.569062 1.157359 6.652514L191.548594 902.717749C152.935158 902.718773 121.521704 871.304296 121.521704 832.691883zM831.792397 902.718773 391.068743 902.718773c0.735757-2.084475 1.157359-4.317329 1.157359-6.652514 0-141.581576 55.054897-274.608312 155.023726-374.578164 95.245248-95.245248 220.499973-149.720953 354.570481-154.655336l0 465.860147C901.819287 871.304296 870.40481 902.718773 831.792397 902.718773z"
              fill="#ffffff" p-id="1590"></path>
              <path d="M349.471346 477.533001c75.04723 0 136.102794-61.054541 136.102794-136.101771s-61.055564-136.102794-136.102794-136.102794-136.102794 61.055564-136.102794 136.102794S274.424116 477.533001 349.471346 477.533001zM349.471346 245.343801c52.982702 0 96.087429 43.104727 96.087429 96.087429 0 52.982702-43.104727 96.087429-96.087429 96.087429-52.982702 0-96.087429-43.104727-96.087429-96.087429C253.383918 288.448528 296.488645 245.343801 349.471346 245.343801z"
              fill="#ffffff" p-id="1591"></path>
          </svg>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as d3 from 'd3';
import * as Global from '../plugins/global';
import GridRender from '../plugins/render_grid';
import ColorGenerater from '../plugins/color_generater';
import { mapState, mapActions } from 'vuex';
import { saveColorResult, saveColorResultZoom } from '../plugins/csv';
import { static_result, static_result2, clear_result, all_result } from '../plugins/evaluate';
window.d3 = d3;

export default {
  name: 'GridView',
  data() {
    return {
      use_image: true,
      allow_zoomout: false,
      allow_details: false,
      show_details: false,
      show_images: true,
      svg_width: 1920,
      svg_height: 1080,
      create_ani: 1000,
      update_ani: 1000,
      remove_ani: 1000,
      mode: 'scan', // 'zoom'
      sample_area: { x1: 0, y1: 0, x2: 0, y2: 0 },
      sample_nodes: [],
      delta: 1e-5,
      in_overview: false,
      mouse_pos: {},
      mouse_pressed: false,
      data_driven: true,
      data_driven_type: 'similar', // 'discri' // 'similar'
      color_part: 'backend', // 'backend' // 'frontend'
      test: false,
      zoom_without_expand: false
    };
  },
  computed: {
    ...mapState(['gridlayout', 'images', 'chosen_ids', 'gridstack', 'colorstack', 'evaluations', 'eval_mode', 'thresholdValue', 'on_loading_flag', 'grid_loading_flag', 'setting_loading_flag']),
    svg: function () {
      return d3.select('.grid-svg');
    },
    grid_group: function () {
      return d3.select('.grid-group');
    },
    overview_group: function () {
      return d3.select('.overview-group');
    },
    confirm_group: function () {
      return d3.select('.confirm-group');
    }
  },
  watch: {
    on_loading_flag: function (on_loading_flag) {
      // document.getElementById('setting-title').innerHTML = 'Settings_' + on_loading_flag;
      if (on_loading_flag > 0) {
        d3.select('#main-gridlayout').classed('pointer-disabled', true);
      } else {
        d3.select('#main-gridlayout').classed('pointer-disabled', false);
      }
    },
    grid_loading_flag: function (grid_loading_flag) {
      if (grid_loading_flag > 0) {
        console.log('show');
        document.getElementById('loading-background').style.visibility = 'visible';
        document.getElementById('loading-svg').style.visibility = 'visible';
        // document.getElementById('loading-svg').style.opacity = '0.7';
        console.log('show end', d3.select('#loading-background-grid').style('display'));
      } else {
        console.log('hide');
        document.getElementById('loading-background').style.visibility = 'hidden';
        document.getElementById('loading-svg').style.visibility = 'hidden';
        console.log('hide end', d3.select('#loading-background-grid').style('display'));
      }
    },
    setting_loading_flag: function (setting_loading_flag) {
      /* if (setting_loading_flag > 0) {
        document.getElementById('loading-background-setting').style.visibility = 'visible';
        document.getElementById('loading-svg-setting').style.visibility = 'visible';
        // d3.select('#loading-background-setting')
        //     .style('display', 'block');
        // d3.select('#loading-svg-setting')
        //     .style('display', 'block');
      } else {
        document.getElementById('loading-background-setting').style.visibility = 'hidden';
        document.getElementById('loading-svg-setting').style.visibility = 'hidden';
        // d3.select('#loading-background-setting')
        //     .style('display', 'none');
        // d3.select('#loading-svg-setting')
        //     .style('display', 'none');
      } */
    },
    evaluations: function (evaluations) {
      if (this.eval_mode === 'top') {
        clear_result();
        console.log('evaluations', evaluations);
        let colorsets = [];
        for (let i = 0; i < evaluations.colors.length; i++) {
          let color = evaluations.colors[i];
          let pcolor = evaluations.pcolors[i];
          Object.keys(color).forEach(key => {
            color[key] = [255 * color[key][0], 255 * color[key][1], 255 * color[key][2]];
          });
          Object.keys(pcolor).forEach(key => {
            pcolor[key] = [255 * pcolor[key][0], 255 * pcolor[key][1], 255 * pcolor[key][2]];
          });
          colorsets.push({
            colors: color,
            pcolors: pcolor
          });
        }
        colorsets.forEach(colorset => {
          this.grid_render.evaluate(colorset, null);
        });

        // let grid_info = evaluations.example;
        // let [meta, grids] = this.grid_render.render(grid_info, null, true);
        // let driven_data = {
        //   grids: grids,
        //   meta: meta
        // };
        // let colormaps = [];
        // for (let i = 0; i < 10; i++) {
        //   colormaps.push(this.color_generater.palettailor(grid_info.color_ids, {}, driven_data, 5 * (i + 1)));
        // }
        // colormaps.forEach(colormap => {
        //   this.grid_render.evaluate(grid_info, colormap);
        // });
        saveColorResult(static_result2, 0, 32, evaluations.times);
      } else {
        console.log('evaluations', evaluations);

        let colorsets = [];
        let cur_grid_id = 0;
        let cur_method_id = 0;
        let max_level = 0;
        for (let i = 0; i < evaluations.colors.length; i++) {
          let color = evaluations.colors[i];
          let pcolor = evaluations.pcolors[i];
          Object.keys(color).forEach(key => {
            color[key] = [255 * color[key][0], 255 * color[key][1], 255 * color[key][2]];
          });
          Object.keys(pcolor).forEach(key => {
            pcolor[key] = [255 * pcolor[key][0], 255 * pcolor[key][1], 255 * pcolor[key][2]];
          });
          let level = evaluations.levels[cur_grid_id];
          colorsets.push({
            colors: color,
            pcolors: pcolor,
            time: evaluations.times[i],
            level: level,
            method: cur_method_id
          });
          cur_method_id += 1;
          if (cur_method_id === 5) {
            cur_method_id = 0;
            cur_grid_id += 1;
          }
          if (level > max_level) max_level = level;
        }

        let mode = 'level';
        let seq = [1, 2, 3, 0, 4];
        if (mode === 'level') {
          clear_result();
          let parts = [];
          let times = [];
          let start = 0;
          let end = 0;
          for (let i = 1; i <= max_level; i++) {
            let tmp = [];
            let colorset = colorsets.filter(colorset => colorset.level === i);
            for (let j = 0; j < 5; j++) {
              let m = seq[j];
              start = end;
              let colorset2 = colorset.filter(colorset => colorset.method === m);
              colorset2.forEach(cs => {
                this.grid_render.evaluate(cs, null);
                end += 1;
                times.push(cs.time);
              });
              tmp.push([start, end]);
            }
            parts.push(tmp);
          }
          saveColorResultZoom(static_result2, parts, times);
        } else {
          clear_result();
          let parts = [];
          let times = [];
          let start = 0;
          let end = 0;
          for (let i = 0; i < 5; i++) {
            let m = seq[i];
            start = end;
            let colorset = colorsets.filter(colorset => colorset.method === m);
            colorset.forEach(cs => {
              this.grid_render.evaluate(cs, null);
              end += 1;
              times.push(cs.time);
            });
            parts.push([start, end]);
          }
          saveColorResultZoom(static_result2, [parts], times);
        }
      }
    },
    gridlayout: async function (grid_info) {
      if (this.in_overview) this.exitOverview();

      if (this.color_part === 'backend') {
        this.grid_render.render(grid_info);
        this.grid_render.evaluate(grid_info);

        if (this.test) {
          let refresh_times = localStorage.getItem('refresh_times');
          if (refresh_times === null) {
            localStorage.setItem('refresh_times', 1);
          } else {
            refresh_times = parseInt(refresh_times);
            console.log('refresh_times', refresh_times);
            if (refresh_times >= 10) {
              localStorage.setItem('refresh_times', 0);
            } else {
              localStorage.setItem('refresh_times', refresh_times + 1);
              setTimeout(() => {
                window.location.reload();
              }, 5000);
            }
          }
        }

        return;
      }

      if (!this.data_driven) {
        let color_set = await this.color_generater.generate_palette(grid_info.color_ids, this.gridstack[this.gridstack.length - 1], this.gridstack[this.gridstack.length - 2]);
        this.grid_render.render(grid_info, color_set);
      } else {
        let [meta, grids] = this.grid_render.render(grid_info, null, true);
        let driven_data = {
          grids: grids,
          meta: meta,
          type: this.data_driven_type
        };
        let color_set = await this.color_generater.generate_palette(grid_info.color_ids, this.gridstack[this.gridstack.length - 1], this.gridstack[this.gridstack.length - 2],
          'simulated_annealing', true, false, driven_data);
        this.grid_render.render(grid_info, color_set, true);
      }
    },
    images: function (images) {
      this.grid_render.render_images(images, this.chosen_ids);
    },
    gridstack: function (gridstack) {
      if (gridstack.length > 2) {
        this.allow_zoomout = true;
      } else {
        this.allow_zoomout = false;
      }
      this.show_details = false;
    }
  },
  mounted() {
    console.log('grid mounted');
    document.getElementById('loading-background-grid').style.visibility = 'hidden';
    document.getElementById('loading-svg-grid').style.visibility = 'hidden';
    window.static_result = static_result;
    window.clear_result = clear_result;
    window.all_result = all_result;
    window.evaluate = () => {
      this.fetchEvaluations();
    };
    let container = d3.select('.gridlayout');
    let bbox = container.node().getBoundingClientRect();
    this.svg_width = bbox.width;
    this.svg_height = bbox.height;
    this.svg.attr('width', this.svg_width);
    this.svg.attr('height', this.svg_height);
    this.create_ani = Global.Animation;
    this.update_ani = Global.Animation;
    this.remove_ani = Global.Animation;

    let that = this;
    this.grid_render = new GridRender(that);
    this.color_generater = new ColorGenerater(that);
    this.initOverview();
  },
  methods: {
    ...mapActions(['fetchGridLayout', 'fetchZoomOutGridLayout', 'fetchImages', 'fetchColorMap', 'pushColormap', 'popColormap', 'fetchEvaluations', 'addOnLoadingFlag', 'decOnLoadingFlag', 'addGridLoadingFlag', 'decGridLoadingFlag']),
    // zoom in overview
    overview_recall_mousedown: function (ev) {
      this.mouse_pos = {
        x: ev.offsetX,
        y: ev.offsetY
      };
      this.mouse_pressed = true;
      this.adjustSamplingArea(this.mouse_pos.x, this.mouse_pos.y, this.mouse_pos.x, this.mouse_pos.y);
      this.confirm_group.style('visibility', 'hidden');
    },
    overview_recall_mousemove: function (ev) {
      if (!this.mouse_pressed) {
        return;
      }
      this.adjustSamplingArea(this.mouse_pos.x, this.mouse_pos.y, ev.offsetX, ev.offsetY);
    },
    overview_recall_mouseup: function (ev) {
      if (!this.mouse_pressed) {
        return;
      }
      this.mouse_pressed = false;
      this.sample_area = {
        x1: this.mouse_pos.x,
        y1: this.mouse_pos.y,
        x2: ev.offsetX,
        y2: ev.offsetY
      };
      if (Math.abs(this.sample_area.x1 - this.sample_area.x2) < this.delta && Math.abs(this.sample_area.y1 - this.sample_area.y2) < this.delta) {
        this.confirm_group.style('visibility', 'hidden');
        return;
      }
      this.confirm_group
        .attr('transform', 'translate(' + (ev.offsetX) + ',' + (ev.offsetY) + ')')
        .style('visibility', 'visible');
    },
    initOverview: function () {
      this.overview_group
        .attr('transform', 'translate(0, 0)')
        .style('visibility', 'hidden');
      this.overview_group
        .append('rect')
        .attr('id', 'overview')
        .attr('class', 'overview-box');
      this.overview_group
        .selectAll('.overview-box')
        .attr('x', 0)
        .attr('y', 0)
        .style('fill', 'white')
        .style('stroke', 'grey')
        .style('stroke-width', 5)
        .style('opacity', 0.3);
      this.overview_group
        .append('rect')
        .attr('id', 'viewbox')
        .style('stroke-dasharray', '5, 5')
        .style('fill', 'white')
        .style('stroke', 'grey')
        .style('stroke-width', 5)
        .style('opacity', 0.5);

      this.overview_group
        .style('pointer-events', 'none');
      // .on('mousedown', this.overview_recall_mousedown)
      // .on('mousemove', this.overview_recall_mousemove)
      // .on('mouseup', this.overview_recall_mouseup);

      this.confirm_group
        .attr('id', 'confirm-zoom')
        .style('visibility', 'hidden')
        .style('cursor', 'pointer')
        .on('click', this.onZoomInClick);
      this.confirm_group.append('circle')
        .attr('r', 20)
        .attr('fill', 'grey');
      let box = this.confirm_group.append('g')
        .attr('class', 'confirm-icon')
        .attr('transform', `scale(${26 / 1024}) translate(${-512}, ${-512})`);
      box.append('path')
        .attr('d', Global.d_confirm1)
        .attr('fill', 'white')
        .attr('stroke', 'black')
        .attr('stroke-width', 2);
      box.append('path')
        .attr('d', Global.d_confirm2)
        .attr('fill', 'white')
        .attr('stroke', 'black')
        .attr('stroke-width', 2);
    },
    enterOverview: function () {
      this.mode = 'zoom';
      this.allow_zoomout = false;
      d3.select('#cropping').select('path').attr('d', Global.d_rollback);
      let meta = this.grid_render.meta;
      this.overview_group.select('#overview')
        .attr('x', meta.delta_x)
        .attr('y', meta.delta_y)
        .attr('width', meta.grid_width)
        .attr('height', meta.grid_height);
      this.overview_group.style('visibility', 'visible');
      this.in_overview = true;
    },
    exitOverview: function () {
      this.mode = 'scan';
      if (this.gridstack.length > 2) this.allow_zoomout = true;
      d3.select('#cropping').select('path').attr('d', Global.d_scan);
      this.overview_group.style('visibility', 'hidden');
      this.in_overview = false;
      this.overview_group.select('#viewbox')
        .attr('width', 0)
        .attr('height', 0);
      this.confirm_group.style('visibility', 'hidden');
    },

    // zoom in sampling
    adjustSamplingArea: function (x1, y1, x2, y2) {
      if (x1 > x2) { let tmp = x1; x1 = x2; x2 = tmp; }
      if (y1 > y2) { let tmp = y1; y1 = y2; y2 = tmp; }
      this.overview_group.select('#viewbox')
        .attr('x', x1)
        .attr('y', y1)
        .attr('width', x2 - x1)
        .attr('height', y2 - y1);
    },
    filterSamples: function () {
      this.sample_nodes = this.grid_render.filter_grids(this.sample_area);
      console.log('zoom in', this.sample_nodes);
    },

    // buttons click apis
    onCropClick: function () {
      if (this.mode === 'scan') {
        this.enterOverview();
      } else {
        this.exitOverview();
      }
    },
    onCrop1Click: function () {
      this.zoom_without_expand = false;
      this.onCropClick();
    },
    onCrop2Click: function () {
      this.zoom_without_expand = true;
      this.onCropClick();
    },
    onZoomInClick: function () {
      // TODO: 根据 zoom_without_expand 是 true/false 调整 fetch_grid_layout的参数即可
      this.filterSamples();
      this.grid_render.in_update = true;
      let args = {
        samples: this.sample_nodes,
        zoom_without_expand: this.zoom_without_expand
      };
      this.fetchGridLayout(args);
    },
    onZoomOutClick: function () {
      this.grid_render.in_update = true;
      this.grid_render.is_zoomout = true;
      this.fetchZoomOutGridLayout();
    },
    onDetailsClick: function () {
      this.show_details = !this.show_details;
      if (this.show_details) this.grid_render.show_details();
      else this.grid_render.hide_details();
    },
    onImageButtonClick: function () {
      this.show_images = !this.show_images;
      if (this.show_images) this.grid_render.show_images();
      else this.grid_render.hide_images();
    }
  }
};
</script>

<style scoped>

.pointer-disabled {
  pointer-events: none;
}

#grid-container {
  width: 67%;
  max-width: calc(98% - 380px);
  margin-left: 10px;
  position: relative;
}

.gridlayout {
  width: 100%;
  height: calc(100% - 23px);
  position: relative;
}

#buttons {
  position: absolute;
  left: 10px;
  top: 10px;
}

.button {
  position: static;
  cursor: pointer;
  overflow: hidden;
  margin-bottom: 20;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  vertical-align: middle;
  color: #fff;
  z-index: 1;
  width: 40px;
  height: 40px;
  line-height: 40px;
  padding: 0;
  background-color: #9e9e9e;
  border-radius: 50%;
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.12), 0 1px 5px 0 rgba(0, 0, 0, 0.2);
}

.gap {
  height: 10px;
}

/* #zoomout {
  position: absolute;
  left: 60px;
  top: 10px;
  cursor: pointer;
  overflow: hidden;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  vertical-align: middle;
  color: #fff;
  z-index: 1;
  width: 40px;
  height: 40px;
  line-height: 40px;
  padding: 0;
  background-color: #9e9e9e;
  border-radius: 50%;
  box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.14),
    0 3px 1px -2px rgba(0, 0, 0, 0.12), 0 1px 5px 0 rgba(0, 0, 0, 0.2);
} */
/*!
 * Waves v0.6.0
 * http://fian.my.id/Waves
 *
 * Copyright 2014 Alfiana E. Sibuea and other contributors
 * Released under the MIT license
 * https://github.com/fians/Waves/blob/master/LICENSE
 */
</style>
