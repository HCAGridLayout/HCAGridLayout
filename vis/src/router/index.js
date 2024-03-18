/* eslint-disable */
import Vue from 'vue';
import Router from 'vue-router';
import GridLayout from '@/components/GridLayout';
import TestDemo from '@/components/Test';

Vue.use(Router);

export default new Router({
  routes: [
    {
      path: '/',
      name: 'GridLayout',
      // component: TestDemo
      component: GridLayout
    }
  ]
});
