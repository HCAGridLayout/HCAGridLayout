/* eslint-disable */
import * as d3 from 'd3';
import * as munsell from 'munsell';

var _norm = function (value, module) {
    return (value % module + module) % module;
};

var _cut_left = function (value, bound) {
    return value < bound ? bound : value;
};

var _cut_right = function (value, bound) {
    return value > bound ? bound : value;
};


var createRandomGenerator = function (rseed) {
    let state = rseed;
    const a = 1664525;
    const c = 1013904223;
    const m = Math.pow(2, 32);
  
    return function() {
      state = (a * state + c) % m;
      return state / m;
    };
};

var rseed = 1234;
var randomGenerator = createRandomGenerator(rseed);

var generateRandomNumberFromRanges = function (ranges) {
    let totalLength = 0;
  
    for (let i = 0; i < ranges.length; i++) {
      let range = ranges[i];
      let rangeLength = range[1] - range[0];
      totalLength += rangeLength;
    }

    if (totalLength === 0) {
      return null;
    }

    let randomNum = randomGenerator() * totalLength;
  
    let accumulatedLength = 0;
    for (let j = 0; j < ranges.length; j++) {
      let range = ranges[j];
      let rangeLength = range[1] - range[0];
  
      if (randomNum >= accumulatedLength && randomNum < accumulatedLength + rangeLength) {
        let randomNumber = randomNum - accumulatedLength + range[0];
        return randomNumber;
      }
  
      accumulatedLength += rangeLength;
    }
    return null;
};

var findIntervalIntersection = function (intervals1, intervals2) {
    let intersection = [];

    for (let i = 0; i < intervals1.length; i++) {
      let interval1 = intervals1[i];
      for (let j = 0; j < intervals2.length; j++) {
        let interval2 = intervals2[j];
        let start = Math.max(interval1[0], interval2[0]);
        let end = Math.min(interval1[1], interval2[1]);
        if (start < end) {
            intersection.push([start, end]);
        }
      }
    }
  
    return intersection;
};

var findIntervalUnion = function (intervals1, intervals2) {
    let mergedIntervals = [...intervals1, ...intervals2];
  
    mergedIntervals.sort((a, b) => a[0] - b[0]);
  
    let union = [mergedIntervals[0]];
  
    for (let i = 1; i < mergedIntervals.length; i++) {
      let currentInterval = mergedIntervals[i];
      let previousInterval = union[union.length - 1];
  
      if (currentInterval[0] <= previousInterval[1]) {
        previousInterval[1] = Math.max(previousInterval[1], currentInterval[1]);
      } else {
        union.push(currentInterval);
      }
    }
  
    return union;
  }
  
const MoonSpencer = function (parent) {
    let that = this;
    that.parent = parent;

    that.sgn_hue = function (delta_hue) {
        let hue = Math.abs(delta_hue);
        if (hue < 0.05) {
            return [1, 'identity']; 
        }
        else if (7 < hue < 12) {
            return [1, 'similarity'];
        }
        else if (28 < hue < 50) {
            return [1, 'contrast'];
        }
        return [0, 'unharmonious'];
    };

    that.sgn_vc = function (delta_v, delta_c) {
        let firE = Math.pow(delta_c / 3, 2) + Math.pow(delta_v / 0.5, 2);
        let secE = Math.pow(delta_c / 5, 2) + Math.pow(delta_v / 1.5, 2);
        let thrE = Math.pow(delta_c / 7, 2) + Math.pow(delta_v / 2.5, 2);
        if (firE < 0.05) {
            return [1, 'identity'];
        }
        else if (firE > 1 && secE < 1) { 
            return [1, 'similarity'];
        }
        else if (thrE > 1) {
            return [1, 'contrast'];
        }
        return [0, 'unharmonious'];
    }

    that.judgeColorHarmony = function (color1, color2) {
        let hvc1 = munsell.rgb255ToMhvc(color1.r, color1.g, color1.b),
            hvc2 = munsell.rgb255ToMhvc(color2.r, color2.g, color2.b);
        let har_h = that.sgn_hue(hvc1[0] - hvc2[0]),
            har_vc = that.sgn_vc(hvc1[1] - hvc2[1], hvc1[2] - hvc2[2]);
        // console.log(har_h, har_vc);
        if (har_h[0] === 0 || har_vc[0] === 0) {
            return false;
        }
        return true;
    };

    that.genHarmonyColor = function (color, type='similar') {
        let hvc_c = munsell.rgb255ToMhvc(color.r, color.g, color.b);
        let hvc_h = hvc_c[0], hvc_v = hvc_c[1], hvc_croma = hvc_c[2];
        let hue = generateRandomNumberFromRanges(that._getHueScope(hvc_h, 7, 12));
        // considering the value and croma easily just like the hue
        let value = generateRandomNumberFromRanges(that._getValueScope(hvc_v, 0.25 * Math.sqrt(2), 1.5));
        let croma = generateRandomNumberFromRanges(that._getCromaScope(hvc_croma, 1.5 * Math.sqrt(2), 5));
        let r = munsell.mhvcToRgb255(hue, value, croma);
        return d3.rgb(r[0], r[1], r[2]);
    };

    that._getHueScope = function (value, delta1, delta2) {
        console.assert(0 <= delta1 < delta2);
        return [[_norm(value - delta2, 100), _norm(value - delta1, 100)], 
            [_norm(value + delta1, 100), _norm(value + delta2, 100)]];
    };

    that._getValueScope = function (value, delta1, delta2) {
        console.assert(0 <= delta1 < delta2);
        return [[_cut_left(value - delta2, 0), _cut_left(value - delta1, 0)],
            [_cut_right(value + delta1, 10), _cut_right(value + delta2, 10)]];
    };

    that._getCromaScope = function (value, delta1, delta2) {
        console.assert(0 <= delta1 < delta2);
        return [[_cut_left(value - delta2, 0), _cut_left(value - delta1, 0)],
            [value + delta1, value + delta2]];
    };

    
    that.genHarmonyColors = function (colors, type='similar') {
        return colors;
    };
}

export default MoonSpencer;