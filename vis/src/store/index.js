/* eslint-disable */
import Vue from "vue";
import Vuex from "vuex";
import axios from "axios";
Vue.use(Vuex);

export default new Vuex.Store({
  state: {
    server_url: BACKEND_BASE_URL,
    gridlayout: {},
    cur_node: -1,
    images: [],
    chosen_ids: null,
    gridstack: [-1],
    colorstack: [-1],
    evaluations: {},
    eval_mode: "top",
    thresholdValue: 0.8,
    on_loading_flag: 0,
    grid_loading_flag: 0,
    setting_loading_flag: 0,
  },
  mutations: {
    setGridLayout(state, gridlayout) {
      state.gridlayout = gridlayout.data;
      state.cur_node = gridlayout.data.index;
      // console.log("set grid layout", gridlayout.data);
    },
    setEvaluations(state, evaluations) {
      state.evaluations = evaluations;
    },
    setImages(state, payload) {
      state.chosen_ids = payload.ids; // may be null
      state.images = payload.images;
    },
    stackPush(state, item) {
      state.gridstack.push(item);
    },
    stackClear(state) {
      state.gridstack = [-1];
    },
    stackPop(state) {
      state.gridstack.pop();
    },
    stackPopTo(state, index) {
      state.gridstack.splice(index, state.gridstack.length - index);
    },
    colorPush(state, item) {
      state.colorstack.push(item);
    },
    colorPop(state) {
      state.colorstack.pop();
    },
    colorClear(state) {
      state.colorstack = [];
    },
    mutSetThresholdValue(state, thresholdValue) {
      state.thresholdValue = thresholdValue;
    },
    mutAddOnLoadingFlag(state) {
      state.on_loading_flag += 1;
    },
    mutDecOnLoadingFlag(state) {
      state.on_loading_flag -= 1;
    },
    mutAddGridLoadingFlag(state) {
      state.grid_loading_flag += 1;
    },
    mutDecGridLoadingFlag(state) {
      state.grid_loading_flag -= 1;
    },
    mutAddSettingLoadingFlag(state) {
      state.setting_loading_flag += 1;
    },
    mutDecSettingLoadingFlag(state) {
      state.setting_loading_flag -= 1;
    },
  },
  getters: {
    gridlayout: state => state.gridlayout
  },
  actions: {
    setThresholdValue({ commit, state }, [thresholdValue]) {
      commit("mutSetThresholdValue", thresholdValue);
    },
    addOnLoadingFlag({ commit, state }) {
      commit("mutAddOnLoadingFlag");
    },
    decOnLoadingFlag({ commit, state }) {
      commit("mutDecOnLoadingFlag");
    },
    addGridLoadingFlag({ commit, state }) {
      commit("mutAddGridLoadingFlag");
      // console.log("add", state.grid_loading_flag);
    },
    decGridLoadingFlag({ commit, state }) {
      commit("mutDecGridLoadingFlag");
      // console.log("dec", state.grid_loading_flag);
    },
    addSettingLoadingFlag({ commit, state }) {
      commit("mutAddSettingLoadingFlag");
    },
    decSettingLoadingFlag({ commit, state }) {
      commit("mutDecSettingLoadingFlag");
    },
    pushColormap({ commit, state }, colormap) {
      commit("colorPush", colormap);
    },
    popColormap({ commit, state }) {
      commit("colorPop");
    },
    async fetchGridLayout({dispatch, commit, state }, args) {
      dispatch('addOnLoadingFlag');
      dispatch('addGridLoadingFlag');
      let tmp_flag = (state.cur_node == -1 ? true : false);
      if (tmp_flag) {
        dispatch('addSettingLoadingFlag');
      }
      // console.log(key, `${state.server_url}/api/gridlayout`)
      let key = {
        node_id: state.cur_node,
        samples: args.samples,
        zoom_without_expand: args.zoom_without_expand
      };
      const gridlayout = await axios.post(
        `${state.server_url}/api/gridlayout`,
        key,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      );
      commit("setGridLayout", gridlayout);
      commit("stackPush", gridlayout.data.index);
      dispatch('decOnLoadingFlag');
      dispatch('decGridLoadingFlag');
      if (tmp_flag) {
        dispatch('decSettingLoadingFlag');
      }
    },
    async fetchZoomOutGridLayout({dispatch, commit, state }) {
      dispatch('addOnLoadingFlag');
      dispatch('addGridLoadingFlag');
      // console.log("zoom out fetch");
      let key = {
        node_id: state.cur_node,
        samples: -1
      };
      const gridlayout = await axios.post(
        `${state.server_url}/api/gridlayout`,
        key,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      );
      // console.log("zoom out fetch done");
      commit("setGridLayout", gridlayout);
      commit("stackPop");
      dispatch('decOnLoadingFlag');
      dispatch('decGridLoadingFlag');
    },
    async fetchImages({dispatch, commit, state }, chosen_info = null) {

      let chosen_names = null, chosen_ids = null;
      if (chosen_info !== null) {
        chosen_ids = chosen_info.ids;
        chosen_names = chosen_info.names;
      }
      if ((chosen_info == null) || (chosen_info.batch)) {
        dispatch('addOnLoadingFlag');
        dispatch('addGridLoadingFlag');
      }

      let key = {
        node_id: state.cur_node
      };
      if (chosen_names !== null) {
        key.chosen_names = chosen_names;
      }
      const images = await axios.post(
        `${state.server_url}/api/instances`,
        key,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      );
      commit("setImages", {
        images: images.data,
        ids: chosen_ids
      });
      if ((chosen_info == null) || (chosen_info.batch)) {
        dispatch('decOnLoadingFlag');
        dispatch('decGridLoadingFlag');
      }
    },
    async fetchColorMap({ commit, state }, colormap = null) {
      let key = { save: false };
      if (colormap !== null) {
        key.save = true;
        key.colormap = colormap;
      }
      const res = await axios.post(`${state.server_url}/api/colormap`, key, {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*"
        }
      });
      if (!key.save && res.data.res) {
        commit("colorPush", res.data.map);
      }
    },
    async fetchEvaluations({ commit, state }) {
      // console.log(key, `${state.server_url}/api/gridlayout`)
      let key = {
        mode: state.eval_mode
      };
      const evaluation = await axios.post(
        `${state.server_url}/api/evaluate_script`,
        key,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      );
      if (key.mode == "gen") return;
      commit("setEvaluations", evaluation.data);
    },
    async resetGridLayout({dispatch, commit, state }, [mode, value]) {
      // console.log(key, `${state.server_url}/api/gridlayout`)
      dispatch('addOnLoadingFlag');
      dispatch('addGridLoadingFlag');
      dispatch('addSettingLoadingFlag');
      // console.log("try", mode, value)
      let key = {
        node_id: state.cur_node,
        mode: mode,
        value: value,
      };
      const gridlayout = await axios.post(
        `${state.server_url}/api/reset-gridlayout`,
        key,
        {
          headers: {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
          }
        }
      );
      commit("setGridLayout", gridlayout);
      if (mode == 0){
        commit("stackClear");}
      else{
        commit("stackPop");
      }
      commit("stackPush", gridlayout.data.index);
      dispatch('decOnLoadingFlag');
      dispatch('decGridLoadingFlag');
      dispatch('decSettingLoadingFlag');
    },
  }
});
