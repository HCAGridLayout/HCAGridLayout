<template>
    <!-- <div> Grid layout</div> -->
  <div class="total-screen">
    <ControlView />
    <GridView />
  </div>
</template>

<script>
/* eslint-disable */ 
import ControlView from './ControlView.vue';
import GridView from './GridView.vue';
import {mapActions} from 'vuex';

export default {
  name: 'GridLayout',
  components: {
    ControlView,
    GridView
  },
  methods: {
    ...mapActions([
      'fetchGridLayout'
    ])
  },
  mounted() {
    this.fetchGridLayout([]);
  }
}
</script>

<style>
.total-screen {
  height: 99%;
}
.view {
  display: inline-block;
  vertical-align: top;
  height:99%;
  margin:0.5% 0 0 0.5%;
}
.divider {
  height: 1px;
  background-color: #ddd;
  overflow: hidden;
}
</style>