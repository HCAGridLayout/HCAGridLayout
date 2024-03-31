/* eslint-disable */
import * as d3 from 'd3';

const l_range = [25, 85];
const c_range = [25, 85];

class iterator {
    constructor(range, iter_num, type) {
        this.range = range;
        this.iter_num = iter_num;
        this.cur_pos = -1;
        this.values = [];

        switch (type) {
            case 'linear':
                let step = (range[1] - range[0]) / (iter_num - 1);
                for (let i = 0; i < iter_num; i++) {
                    this.values.push(range[0] + step * i);
                }
                break;
            case 'random':
                for (let i = 0; i < iter_num; i++) {
                    this.values.push(range[0] + (range[1] - range[0]) * Math.random());
                }
                break;
            case 'cos':
                let step_cos = (range[1] - range[0]) / 2;
                for (let i = 0; i < iter_num; i++) {
                    this.values.push(range[0] + step_cos * (1 - Math.cos(2 * Math.PI * i / (iter_num - 1))));
                }
                break;
            case 'quad':
                console.assert(iter_num > 1);
                let arg1 = range[1];
                let arg2 = (range[0] - range[1]) / Math.pow((iter_num - 1) / 2, 2);
                for (let i = 0; i < iter_num; i++) {
                    this.values.push(arg1 + arg2 * Math.pow(i - (iter_num - 1) / 2, 2));
                }
            default:
                let default_value = (range[1] + range[0]) / 2;
                for (let i = 0; i < iter_num; i++) {
                    this.values.push(default_value);
                }
                break;
        }
    }

    next() {
        this.cur_pos++;
        // if (this.cur_pos >= this.iter_num) {
        //     return null;
        // }
        return this.values[this.cur_pos];
    }

}

function extend_in_same_hue(color_hex, palette_size) {
    let color = d3.hcl(color_hex);
    let colors = [];
    let range_l = [Math.max(l_range[0], l_range[1] - 10 * palette_size), l_range[1]];
    let range_c = [Math.max(c_range[0], c_range[1] - 12 * palette_size), c_range[1]];
    let l_iter = new iterator(range_l, palette_size, 'linear');
    let c_iter = new iterator(range_c, palette_size, 'quad');
    for (let i = 0; i < palette_size; i++) {
        let tp_color = d3.hcl(color.h, c_iter.next(), l_iter.next());
        colors.push(d3.rgb(tp_color));
        // console.log(tp_color);
    }
    return colors;
}

export {
    extend_in_same_hue
}
