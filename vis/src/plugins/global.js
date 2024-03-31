/* eslint-disable */
import * as d3 from "d3";

const Animation = 500;
const QuickAnimation = 400;
const logic_height = 1080;
const logic_width = 1920;

const d_rollback = "M793 242H366v-74c0-6.7-7.7-10.4-12.9-6.3l-142 112c-4.1 3.2-4.1 9.4 0 12.6l142 112c5.2 4.1 12.9 0.4 12.9-6.3v-74h415v470H175c-4.4 0-8 3.6-8 8v60c0 4.4 3.6 8 8 8h618c35.3 0 64-28.7 64-64V306c0-35.3-28.7-64-64-64z";
const d_scan = "M477.663 848v64h-177v-64h177z m-235 0v64h-39.858C153.759 912 114 872.24 114 823.195v-48.044h64v48.044c0 13.7 11.105 24.805 24.805 24.805h39.858zM178 717.15h-64v-177h64v177z m0-235h-64v-177h64v177z m0-235h-64v-43.345C114 154.759 153.76 115 202.805 115h44.556v64h-44.556c-13.7 0-24.805 11.105-24.805 24.805v43.346zM305.36 179v-64h177v64h-177z m235 0v-64h177v64h-177z m235 0v-64h46.835C871.241 115 911 154.76 911 203.805v41.068h-64v-41.068c0-13.7-11.105-24.805-24.805-24.805h-46.834zM847 302.873h64v174.962h-64V302.873z m-57.059 439.485l112.271 112.271c12.497 12.497 12.497 32.758 0 45.255l-2.828 2.828c-12.497 12.497-32.758 12.497-45.255 0l-113-113L692.01 912.3 571.095 573.595 909.8 694.51l-119.858 47.848z";
const d_confirm1 = "M949.9 899l-124-124.1c28.2-33.3 51.1-70.4 68.2-110.8 22.5-53.3 34-109.9 34-168.2s-11.4-114.9-34-168.2c-21.8-51.5-52.9-97.6-92.6-137.3-39.7-39.7-85.8-70.8-137.3-92.6C610.9 75.4 554.3 64 496 64S381.1 75.4 327.8 98c-51.5 21.8-97.6 52.9-137.3 92.6-39.7 39.7-70.8 85.8-92.6 137.3C75.4 381.1 64 437.7 64 496s11.4 114.9 34 168.2c21.8 51.5 52.9 97.6 92.6 137.3 39.7 39.7 85.8 70.8 137.3 92.6 53.3 22.5 109.9 34 168.2 34s114.9-11.4 168.2-34c40.4-17.1 77.5-40 110.8-68.2l123.9 124c14 14 36.9 14 50.9 0s14-36.9 0-50.9zM496 856c-198.8 0-360-161.2-360-360s161.2-360 360-360 360 161.2 360 360-161.2 360-360 360z";
const d_confirm2 = "M644 460H532V348c0-19.8-16.2-36-36-36s-36 16.2-36 36v112H348c-19.8 0-36 16.2-36 36s16.2 36 36 36h112v112c0 19.8 16.2 36 36 36s36-16.2 36-36V532h112c19.8 0 36-16.2 36-36s-16.2-36-36-36z";

const deepCopy = function (obj) {
    let _obj = Array.isArray(obj) ? [] : {}
    for (let i in obj) {
        _obj[i] = typeof obj[i] === 'object' ? deepCopy(obj[i]) : obj[i]
    }
    return _obj
};

function pos2str_inverse(pos) {
    return pos.y + "," + pos.x;
}

function pos2str(pos) {
    return pos.x + "," + pos.y;
}


const getTextWidth = function (text, font) {
    let canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
    let context = canvas.getContext("2d");
    context.font = font;
    return context.measureText(text).width;
}

function disable_global_interaction() {
    d3.select(".loading")
        .style("display", "block")
        .style("opacity", 0.5);
}

function enable_global_interaction(delay) {
    delay = delay || 1;
    d3.select(".loading")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none")
        .style("opacity", 1);

}

function begin_loading() {
    // $(".loading").show();
    // $(".loading-svg").show();
    d3.select(".loading")
        .style("display", "block");
    d3.select(".loading-svg")
        .style("display", "block");
}

function end_loading(delay) {
    delay = delay || 1;
    // console.log("delay", delay);
    d3.select(".loading")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none");
    d3.select(".loading-svg")
        .transition()
        .duration(1)
        .delay(delay)
        .style("display", "none");
}

export {
    Animation,
    QuickAnimation,
    logic_height,
    logic_width,
    d_rollback,
    d_scan,
    d_confirm1,
    d_confirm2,
    deepCopy,
    pos2str,
    pos2str_inverse,
    getTextWidth,
    disable_global_interaction,
    enable_global_interaction,
    begin_loading,
    end_loading
}
