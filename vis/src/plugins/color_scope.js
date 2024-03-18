import * as d3 from 'd3';
const colorscope = {
    'hue_scope': [0, 360],
    'chroma_scope': [20, 80],
    'lumi_scope': [35, 80],
    'chroma_plus_lumi': [55, 155]
};

class ColorScope {
  constructor(scope = colorscope) {
    this.scope = scope;
  }

  disturbColor(color, randf, randg) { 
    // randf: random function, randg: random scope [3]
    let hcl = d3.hcl(color);
    let hue = hcl.h;
    let chroma = randf(Math.max(Math.round(hcl.c - randg[1]), this.scope['chroma_scope'][0]),
        Math.min(Math.round(hcl.c + randg[1]), this.scope['chroma_scope'][1]));
    let lumi = randf(Math.max(Math.round(hcl.l - randg[2]), this.scope['lumi_scope'][0]),
        Math.min(Math.round(hcl.l + randg[2]), this.scope['lumi_scope'][1]));
    
    if (this.scope['hue_scope'][0] === 0 && this.scope['hue_scope'][1] === 360) {
      hue = hcl.h + randf(-randg[0], randg[0]);
      hue = hue % 360 + (hue < 0 ? 360 : 0);
    } else {
      hue = randf(Math.max(Math.round(hcl.h - randg[0]), this.scope['hue_scope'][0]),
      Math.min(Math.round(hcl.h + randg[0]), this.scope['hue_scope'][1]));
    }

    if ((chroma + lumi) > this.scope['chroma_plus_lumi'][1]) {
      let margin = (chroma + lumi - this.scope['chroma_plus_lumi'][1]) / 2;
      chroma -= margin;
      lumi -= margin;
    }
    return d3.hcl(hue, chroma, lumi);
  }
}

export default ColorScope;
