<template>
  <div class="view" id="control-container">
    <div class="sub-control-outter">
      <p id="setting-title" class="sub-title" style="color: #4e4e4e;">Settings</p>
      <div class="sub-control">
        <div id='loading-background-setting' class="my-loading" style="display: block;"></div>
        <svg id='loading-svg-setting' version="1.1" width="40px" height="40px" style="enable-background:new 0 0 512 512; display: block;" xml:space="preserve"
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
        <v-container style="height: 100%;">
          <v-row justify="center" align="center" style="height: 25%;">
            <v-col cols="3" class="V-centered" style="height: 100%;">
              <span class="label-text">Dataset:</span>
            </v-col>
            <v-col cols="6" style="height: 100%;">
              <v-select
                v-model="selectedOption"
                :items="options"
                @change="updateOptions2"
                solo
                dense>
              </v-select>
            </v-col>
            <v-col cols="3" class="V-centered H-centered" style="height: 100%;">
              <el-tooltip :content="layout_type" placement="top">
                <el-switch v-model="use_HV" active-color="#aaaaaa" inactive-color="#aaaaaa" @change="changeSwitch"></el-switch>
              </el-tooltip>
            </v-col>
          </v-row>
          <v-row justify="center" align="center" style="height: 25%;">
            <v-col cols="3" class="V-centered" style="height: 100%;">
              <span class="label-text">Model:</span>
            </v-col>
            <v-col cols="6" style="height: 100%;">
              <v-select
                v-model="selectedOption2"
                :items="options2"
                solo
                dense
              ></v-select>
            </v-col>
            <v-col cols="3" style="height: 100%;" class="H-centered">
              <v-btn color="#aaaaaa" style="text-transform: none; color: white; width: 100%; !important" @click="loadButtonClicked">Load</v-btn>
            </v-col>
          </v-row>
          <v-row class="mt-4" justify="center" align="center" style="height: 15%;">
            <v-col cols="9" style="height: 100%;">
              <div style="display: inline-block;" class="V-centered"> <span class="label-text">Ambiguity threshold:</span> </div>
            </v-col>
            <v-col cols="3" style="height: 100%;">
              <!-- <div style="display: inline-block;" id="threshold-value-text" class="label-text V-centered">0.8</div> -->
            </v-col>
          </v-row>
          <v-row justify="center" align="center" style="height: 20%;">
            <v-col cols="9" style="height: 100%;">
              <v-slider
                v-model="sliderValue"
                color="#aaaaaa"
                track-color="#aaaaaa"
                min="0"
                max="10"
                thumb-label
                thumb-color="#505050"
                @change="handleSliderChange">
                <template v-slot:thumb-label>
                  <v-row justify="center">
                    <v-col cols="auto">
                      <span>{{ selectedLabel }}</span>
                    </v-col>
                  </v-row>
                </template>
              </v-slider>
            </v-col>
            <v-col cols="3" class="H-centered" style="height: 100%;">
              <v-btn color="#aaaaaa" style="text-transform: none; color: white; width: 100%; !important" @click="layoutButtonClicked">Layout</v-btn>
            </v-col>
          </v-row>
        </v-container>
      </div>
    </div>

    <div class="sub-control-outter" style="height: 40%">
      <p class="sub-title" style="color: #4e4e4e;">Class</p>
      <div class="sub-control">
        <svg class="class-svg" id="class-tree" style="width: 100%; height: 90%;">
        <g/>
        </svg>
      </div>
    </div>

    <div class="sub-control-outter" style="height: 30%">
      <p class="sub-title" style="color: #4e4e4e;">Sample</p>
      <div class="sub-control">
        <v-row style="height: 90%" justify="center" align="center">
          <v-col cols="4" style="height: 100%">
            <div class="meta-info">
              <label style="font-weight: bold;">Prediction: </label>
              <label class="meta-label" id="meta1"></label>
              <label class="meta-label" id="meta1t" style="display: block;"></label>
            </div>
            <div class="meta-info">
              <label style="font-weight: bold;">Ground Truth: </label>
              <label class="meta-label" id="meta2"></label>
              <label class="meta-label" id="meta2t" style="display: block;"></label>
            </div>
            <div class="meta-info">
              <label style="font-weight: bold;">Prob. Distribution: </label>
              <!--   <label class="meta-label" style="display: block;" id="meta3"></label> -->
            </div>
            <div style="height: 3%;"/>
            <svg class="datainfo-bar" id="meta-bar" preserveAspectRatio="none">
              <g>
                <rect class="bar-boundary" fill="rgba(127, 127, 127)"/>
              </g>
            </svg>
          </v-col>
          <v-col cols="8" style="height: 100%">
            <div class="image-info H-centered V-centered" id="meta-image-container">
              <img class="datainfo-image" id="meta-image"/>
            </div>
          </v-col>
        </v-row>
      </div>
    </div>

  </div>
</template>

<script>
/* eslint-disable */
import * as d3 from 'd3';
import { mapState, mapActions } from 'vuex';
export default {
  computed: {
  ...mapState(['gridlayout', 'thresholdValue']),
    class_svg: function () {
      return d3.select('.class-svg');
    },
    selectedLabel() {
      return this.slider_values_threshold_str[this.sliderValue];
    }
  },
  name: 'ControlView',
  data() {
    return {
      msg: 'Welcome to Your Vue.js App',
      selectedOption: 'ImageNet1k Animals',
      options: ['ImageNet1k Animals', 'Cifar100'],
      selectedOption2: 'VITb',
      options2: ['VITb'],
      DatasetModels: {
        'Cifar100': {'model_name2': "cifar100"},
        'ImageNet1k Animals': {'VITb': "imagenet1k_animals"}
      },
      use_HV: false,
      layout_type : "Power Diagram Layout",
      sliderValue: 8,
      //alphaValue: 1/2
      // slider_values_alpha: [2, 1, 1/2, 1/4, 1/6, 1/8, 1/12, 1/16, 1/24, 1/32],
      // slider_values_alpha_str: ["2", "1", "1/2", "1/4", "1/6", "1/8", "1/12", "1/16", "1/24", "1/32"],
      slider_values_threshold: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
      slider_values_threshold_str: ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"],
    }
  },
  watch: {
    gridlayout: function(grid_info) {
      for(let i=0;i<this.slider_values_threshold.length;i++){
        if(Math.abs(this.slider_values_threshold[i]-grid_info.ambiguity_threshold)<Math.abs(this.slider_values_threshold[this.sliderValue]-grid_info.ambiguity_threshold))
          this.sliderValue = i;
      }
      this.handleSliderChange(this.sliderValue);

      for(let key1 in this.DatasetModels) {
        for(let key2 in this.DatasetModels[key1]) {
          if(this.DatasetModels[key1][key2] == grid_info.dataset_name) {
            this.selectedOption = key1;
            this.updateOptions2(key1);
            this.selectedOption2 = key2;
            // console.log("key1", key1, "key2", key2);
            break;
          }
        }
      }

      this.use_HV = grid_info.info_dict.use_HV;
      this.changeSwitch();

      this.render_class(grid_info);
    }
  },
  mounted() {
    // console.log('control mounted');
    document.getElementById('loading-background-setting').style.visibility = 'hidden';
    document.getElementById('loading-svg-setting').style.visibility = 'hidden';
  },
  methods: {
    ...mapActions(['resetGridLayout', 'setThresholdValue', 'addOnLoadingFlag', 'decOnLoadingFlag']),
    render_class: function(grid_info) {
      // for test
      // console.log(grid_info);

      let label_father = {}
      let top_label_colors = {}
      let label_children = {}
      for(let i=0;i<grid_info.labels.length;i++){
        label_father[grid_info.labels[i]] = grid_info.top_labels[i]
      }
      for(let key in label_father) {
        if (label_father.hasOwnProperty(key)) {
          let key2 = label_father[key];
          if(!label_children.hasOwnProperty(key2)){
            label_children[key2] = [];
            top_label_colors[key2] = grid_info.pcolors[key];
          }
          label_children[key2].push(key);
        }
      }
      // console.log(label_father, label_children);

      let label_data = [];
      let svgElement = document.getElementById('class-tree');
      let svgWidth = svgElement.clientWidth;
      let svgHeight = svgElement.clientHeight;
      svgElement.setAttribute('viewBox', `0 0 ${svgWidth} ${svgHeight}`);

      let label_nums = Object.keys(label_father).length;
      let rect_width = svgWidth*3/7;
      let rect_height = min(30, svgHeight/(label_nums+1/3)*2/3);
      let rect_gap = (svgHeight-rect_height*label_nums)/(label_nums+1);

      const positions = {};
      const tree_data = [];

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

      let bias = rect_gap;
      for(let key2 in label_children) {
        // console.log("key2", key2);
        if(label_children.hasOwnProperty(key2)) {
          let last_bias = bias;
          for(let i=0;i<label_children[key2].length;i++) {
            let key = label_children[key2][i];
            // console.log("key", key);
            let label_id = key;
            if(key==key2)label_id += "_child";
            // let theight = min(rect_height, max(10, rect_height));
            let theight = rect_height*3/4;
            label_data.push({"label_id": label_id, "name": extractString(grid_info.label_names[key]), "color": grid_info.colors[key], "x": svgWidth*6/11, "y": bias, "tx": rect_width/2, "twidth": rect_width*3/4, "theight": theight, "ta": "middle"});
            positions[label_id] = [svgWidth*6/11, bias+rect_height/2];
            label_children[key2][i] = label_id;
            bias += rect_height+rect_gap;
            tree_data.push([key2, label_id]);
          }
          // let theight = min(rect_height, max(10, rect_height));
          let theight = rect_height*3/4;
          label_data.push({"label_id": key2, "name": extractString(grid_info.label_names[key2]), "color": top_label_colors[key2], "x": svgWidth*5/11-rect_width, "y": (last_bias+bias-rect_gap-rect_height)/2, "tx": rect_width/2, "twidth": rect_width*3/4, "theight": theight, "ta": "middle"});
          positions[key2] = [svgWidth*5/11, (last_bias+bias-rect_gap)/2];
        }
      }

      // console.log(label_data);
      // console.log(tree_data, positions);

      let tree_svg = this.class_svg.select('g');
      let update = tree_svg.selectAll(".node")
        .data(label_data, d => d.label_id);

      setTimeout(function() {
        let group = update.enter().append("g")
          .attr("class", "node")
          .attr("transform", d => `translate(${d.x}, ${d.y})`);

        group.append("rect")
          .attr("width", rect_width)
          .attr("height", rect_height)
          .attr("fill", d => `rgba(${d.color[0]*255},${d.color[1]*255},${d.color[2]*255})`)
          .style("opacity", 0)
          .transition()
          .duration(500)
          .style("opacity", 1);

        let text_group = group.append("text")
          .attr("class", "label")
          .attr("x", d => d.tx)
          .attr("y", rect_height/2)
          .attr("text-anchor", d => d.ta)
          .attr("dominant-baseline", "middle")
          .attr("fill", "white")
          .attr("width", d => d.twidth)
          .attr("height", rect_height)
          .text(d => d.name)
          .style("font-size", d => d.theight)
          .style("opacity", 0);

        text_group.each(function(d, i) {
          let text_node = d3.select(this);
          let node_width = text_node.node().getComputedTextLength();
          if(node_width > rect_width*5/6){
            let truncatedText = d.name.substring(0, Math.floor(d.name.length * (rect_width*5/6 / node_width)));
            truncatedText = truncatedText.slice(0, -3) + "...";
            text_node.text(truncatedText);
          }
        });

        text_group.append("title")
          .text(d => d.name);
        text_group.transition()
          .duration(500)
          .style("opacity", 1);
      }, 1000);

      update.exit()
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();

      update.transition()
        .duration(500)
        .delay(500)
        .attr("transform", d => `translate(${d.x}, ${d.y})`)
      update.select("rect")
        .transition()
        .duration(500)
        .delay(500)
        .attr("width", rect_width)
        .attr("height", rect_height)
        .attr("fill", d => `rgba(${d.color[0]*255},${d.color[1]*255},${d.color[2]*255})`);

      update.select("text").each(function(d, i) {
        let text_node = d3.select(this);
        text_node.style("font-size", d.theight);
        text_node.text(d.name);
        let node_width = text_node.node().getComputedTextLength();
        if(node_width > rect_width*5/6){
          let truncatedText = d.name.substring(0, Math.floor(d.name.length * (rect_width*5/6 / node_width)));
          truncatedText = truncatedText.slice(0, -3) + "...";
          text_node.text(truncatedText);
        }

        update.select("text").append("title")
          .text(d => d.name);
      });

      update.select("text")
        .transition()
        .duration(500)
        .delay(500)
        .attr("x", d => d.tx)
        .attr("y", rect_height/2)
        .attr("text-anchor", d => d.ta)
        .attr("dominant-baseline", "middle")
        .attr("fill", "white")
        .attr("width", d => d.twidth)
        .attr("height", rect_height);

      let lineGenerator = d3.line().curve(d3.curveBasis);

      tree_svg.selectAll("path.link")
        .transition()
        .duration(500)
        .style("opacity", 0);

      setTimeout(function() {
        let update2 = tree_svg.selectAll("path.link").data(tree_data);

        update2.exit().remove();

        update2.enter()
          .append("path")
          .attr("class", "link")
          .attr("d", d => {
            const source = positions[d[0]];
            const target = positions[d[1]];
            return lineGenerator([source, [(source[0]+target[0])/2, source[1]], [(source[0]+target[0])/2, target[1]], target]);
          })
          .attr("stroke", "black")
          .attr("fill", "none")
          .style("opacity", 0)
          .transition()
          .duration(500)
          .style("opacity", 1);

        update2.attr("d", d => {
            const source = positions[d[0]];
            const target = positions[d[1]];
            return lineGenerator([source, [(source[0]+target[0])/2, source[1]], [(source[0]+target[0])/2, target[1]], target]);
          })
          .attr("stroke", "black")
          .attr("fill", "none")
          .style("opacity", 0)
          .transition()
          .duration(500)
          .style("opacity", 1);
      }, 1000);
    },
    updateOptions2() {
      this.options2 = Object.keys(this.DatasetModels[this.selectedOption]) || [];
      this.selectedOption2 = null;
      if(this.options2.length>0)this.selectedOption2 = this.options2[0];
    },
    changeSwitch() {
      // console.log("switch", this.use_HV);
      this.layout_type = this.use_HV ? "Treemap Layout" : "Power Diagram Layout";
      // this.addOnLoadingFlag();
    },
    loadButtonClicked() {
      // console.log(this.selectedOption, this.selectedOption2, this.DatasetModels[this.selectedOption][this.selectedOption2]);
      let data_name = this.DatasetModels[this.selectedOption][this.selectedOption2];
      let update_info = {"use_HV": this.use_HV};
      this.resetGridLayout([0, [data_name, update_info]]);
    },
    layoutButtonClicked() {
      // console.log(this.thresholdValue);
      this.resetGridLayout([1, this.thresholdValue]);
    },
    handleSliderChange(value) {
      this.setThresholdValue([this.slider_values_threshold[value]]);
      // document.getElementById("threshold-value-text").innerHTML = this.slider_values_threshold_str[value];
    }
  }
}
</script>


<style scoped>
.sub-title {
  color: rgb(204, 204, 204);
  font-weight: bold;
  font-size: 24px;
}

.label-text {
  color: rgb(78, 78, 78);
  font-weight: bold;
  font-size: 18px;
}


#control-container {
  position: relative;
  width: 30%;
  min-width: 380px;
  border-right: 1px solid #ddd;
}

.sub-control-outter {
  position: relative;
  width: 100%;
  height: 30%;
}

.sub-control {
  position: relative;
  width: 100%;
  height: 90%;
  min-height: 30px;
  line-height: 30px;
  padding-left: 10px;
  padding-right: 10px;
  border-top: 1px solid #ddd;
}

.meta-info {
  width: 100%;
}

.image-info {
  height: 100%;
  width: 100%;
}

.datainfo-image {
  width: auto;
  height: auto;
  max-width: 100%;
  max-height: 100%;
}

.datainfo-bar {
  height: 10%;
  width: 90%;
}

.meta-label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.V-centered {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.H-centered {
  align-items: center;
  text-align: center;
}
</style>
