/* eslint-disable */
// https://observablehq.com/@esperanc/color-harmonies
import * as d3 from 'd3';
const bins = 72;
const hueShift = 0;
const saturation = 100;
const luminosity_hcl = 70;
const luminosity_hsl = 60;
const chroma = 100;
const mode = 'hcl';
const showTemplate = true;

function wheelColors(
    n = bins,
    hs = hueShift,
  ) {
    const hueShiftAngle = (360 / n) * hs;
    return d3
      .range(n)
      .map((i) => {
        if (mode === 'hcl')
          return d3.rgb(d3.hcl((i / n) * 360 + hueShiftAngle, chroma, luminosity_hcl));
        else if (mode === 'hsl') 
          return `hsl(${(i / n) * 360 + hueShiftAngle},${saturation}%,${luminosity_hsl}%)`;
      });
}

function colorWheel (group, n = bins, options = {}) {
    let {
      margin = 20,
      innerRadius = 100,
      outerRadius = 200,
      padAngle = 0,
      padRadius = 100,
      cornerRadius = 0
    } = options;
  
    let arcGenerator = d3
      .arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius)
      .padAngle(padAngle)
      .padRadius(padRadius)
      .cornerRadius(cornerRadius);
  
    let wc = wheelColors();
  
    let arcData = d3.range(n).map((i) => ({
      startAngle: ((i - 0.5) / n) * Math.PI * 2 + Math.PI / 2,
      endAngle: ((i + 0.5) / n) * Math.PI * 2 + Math.PI / 2,
      color: wc[i]
    }));
  
    let img = '.color-wheel-group';
  
    d3.select(img)
      .append("g")
      .attr(
        "transform",
        `translate(${margin + outerRadius},${margin + outerRadius})`
      )
      .selectAll("path")
      .data(arcData)
      .enter()
      .append("path")
      .attr("d", arcGenerator)
      .attr("fill", (d) => d.color);
    return img;
}
const beta = 2 * Math.PI / 360;
function drawTemplate (group, template) {
    // console.log(template);
    group.append('g').attr('class', 'template-group');
    template.ranges.forEach(range => {
        let start = [Math.cos(range[0] * beta), Math.sin(range[0] * beta)];
        let angle = range[1] > range[0] ? range[1] - range[0] : range[1] + 360 - range[0];
        let end = [Math.cos(range[1] * beta), Math.sin(range[1] * beta)];
        let is_large = angle > 180 ? 1 : 0;
        let r = 200;
        let center = [r+20, r+20];
        let path = `M${center[0]},${center[1]}L${start[0]*r+center[0]},${start[1]*r+center[1]}A ${r} ${r} ${0} ${is_large} 1 ${end[0]*r+center[0]}, ${end[1]*r+center[1]} Z`;
        group.append('path')
            .attr('d', path)
            .attr('stroke', 'gray')
            .attr('stroke-width', 3)
            .attr('fill', 'gray')
            .attr('fill-opacity', 0.01);

    });
}

function drawPtrColors (group, colors) {
    group.append('g').attr('class', 'ptr-colors-group');
    colors.forEach((color, i) => {
        let h = 0
        if (mode === 'hcl') h = d3.hcl(color).h;
        else if (mode === 'hsl') h = d3.hsl(color).h;
        // console.log('h', color, h)
        let r = 200;
        let center = [r+20, r+20];
        let ptr_range = [10, 30];
        let angle = [Math.cos(h * beta), Math.sin(h * beta)];
        let path = `M${angle[0]*(r+ptr_range[0])+center[0]},${angle[1]*(r+ptr_range[0])+center[1]}L${angle[0]*(r+ptr_range[1])+center[0]},${angle[1]*(r+ptr_range[1])+center[1]}`;
        group.append('path')
            .attr('d', path)
            .attr('stroke', d3.rgb(color))
            .attr('stroke-width', 5)
            .attr('fill', color);
    });

}

const drawColorWheel = function (group, template = null, ptr_colors = null) {
    let colorWheelGroup = group.append('g').attr('class', 'color-wheel-group');
    colorWheel(colorWheelGroup);
    if (template && showTemplate) {
        drawTemplate(colorWheelGroup, template);
    }
    if (ptr_colors) {
        drawPtrColors(colorWheelGroup, ptr_colors);
    }
    return colorWheelGroup;
}

export default drawColorWheel;