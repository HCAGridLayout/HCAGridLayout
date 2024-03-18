<template>
  <div id="app" data-app>
    <router-view/>
  </div>
</template>

<script>
export default {
  name: 'App'
};
</script>

<style>
html, body, #app {
  width: 100%;
  height:100%;
  margin: 0 0 0 0;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  line-height: 1.42857143;
  color: #333333;
  background-color: #fff;
}
.loading {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: gray;
    opacity: 0.4;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 98;
}
.loading-svg {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 99;
}

.my-loading {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: gray;
    opacity: 0.4;
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 98;
}
.my-loading-svg {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 99;
}

.circle-path {
  transform: translate(15px,15px);
/*   transform-origin: center; */
  animation: rotate_arrow 2s infinite;
}
@keyframes rotate_arrow {
  from {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
    99.999% {
        -webkit-transform: rotate(359deg);
        transform: rotate(359deg);
    }
    to {
        -webkit-transform: rotate(0deg);
        transform: rotate(0deg);
    }
}

</style>
