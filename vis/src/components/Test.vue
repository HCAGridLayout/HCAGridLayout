<template>
    <div class="view" id="control-container">
        <div>Test view</div>
        <svg class='test-svg'>
          <g class='test-g'>
          </g>
        </svg>
    </div>
</template>

<script>
import * as d3 from 'd3';
import {simulatedAnnealing2FindBestPalette, evaluatePalette} from '../plugins/optimizeFunc';
// import {extend_in_same_hue} from '../plugins/hue_extension';
import MoonSpencer from '../plugins/moon_spencer';
import drawColorWheel from '../plugins/color_wheel';
export default {
  name: 'TestDemo',
  data () {
    return {
      msg: 'This is just a test demo',
      base_color: '#ffffff',
      extend_colors: ['#ff0000', '#00ff00', '#0000ff'],
      cases: [
        {
          base_color: '#abe3f6',
          child_num: 5,
          exclude_colors: ['#f8b6bb', '#b9f3be', '#f3e4bf', '#d8c1f6']
        },
        {
          base_color: '#f8b6bb',
          child_num: 4,
          exclude_colors: ['#abe3f6', '#b9f3be', '#f3e4bf', '#d8c1f6']
        },
        {
          base_color: '#b9f3be',
          child_num: 3,
          exclude_colors: ['#abe3f6', '#f8b6bb', '#f3e4bf', '#d8c1f6']
        },
        {
          base_color: '#f3e4bf',
          child_num: 20,
          exclude_colors: ['#abe3f6', '#f8b6bb', '#b9f3be', '#d8c1f6']
        },
        {
          base_color: '#d8c1f6',
          child_num: 3,
          exclude_colors: ['#abe3f6', '#f8b6bb', '#b9f3be', '#f3e4bf']
        },
        {
          base_color: '#f3e4bf',
          child_num: 10,
          exclude_colors: ['#abe3f6', '#f8b6bb', '#b9f3be', '#d8c1f6']
        }
      ],
      colorsscope: {'hue_scope': [0, 360], 'lumi_scope': [0, 40]}
    };
  },
  computed: {
    svg: function () {
      return d3.select('.test-svg');
    },
    g: function () {
      return d3.select('.test-g');
    }
  },
  methods: {
    render: function () {
      this.svg.attr('width', 1200)
        .attr('height', 1000);
      this.svg.append('rect')
        .attr('x', 20)
        .attr('y', 20)
        .attr('width', 60)
        .attr('height', 60)
        .attr('stroke', 'black')
        .attr('stroke-width', 1)
        .attr('fill', this.base_color);
      this.g.selectAll('rect.extend-color-rect')
        .data(this.extend_colors)
        .enter()
        .append('rect')
        .attr('class', 'extend-color-rect')
        .attr('x', 130)
        .attr('y', function (d, i) {
          return i * 80 + 20;
        })
        .attr('width', 60)
        .attr('height', 60)
        .attr('stroke', 'black')
        .attr('stroke-width', 1)
        .attr('fill', function (d) {
          return `rgb(${d.r}, ${d.g}, ${d.b})`;
        });
        drawColorWheel(this.svg.append('g').attr('transform', 'translate(600, 10)'), this.template, this.extend_colors);
    },
    extendColors: function (i) {
      let base_color = this.cases[i].base_color;
      let child_num = this.cases[i].child_num;
      this.base_color = base_color;
      // this.extend_colors = extend_in_same_hue(base_color, child_num);
      // this.moon_spencer.judgeColorHarmony(d3.rgb(base_color), d3.rgb(exclude_colors[1]));
      const best_colors = simulatedAnnealing2FindBestPalette(base_color, child_num, (newpalette) => evaluatePalette(newpalette), this.colorsscope);
      this.extend_colors = best_colors.id;
      this.template = best_colors.harmony_info[2].rotate(best_colors.harmony_info[3]);
    }
  },
  mounted: function () {
    this.moon_spencer = new MoonSpencer();
    this.extendColors(5);
    this.render();
  }
};
</script>

<style scoped>
#control-container {
    width: 23%;
    min-width: 380px;
    border-right:1px solid #ddd;
}
.sub-control {
    height: 30%;
    min-height: 30px;
    line-height: 30px;
    padding-left: 10px;
    border-top: 1px solid #ddd;
}
</style>
