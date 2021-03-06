`use strict`

  //calling function at page load
window.onload = function() {
  var tempOptions = {
    chart: {
      type: 'area',
      height: 300,
      width: 500,
    },
    series: [{
      name: 'Temperature °C',
      data: {{contextos.temperature|safe}}
    }],
    xaxis: {
      categories: {{contextos.data|safe}}
    },
      title: {
      text: 'Temperature °C',
      align: 'left'
    },
  }

  //second chart
  var rainOptions = {
    chart: {
      type: 'area',
      height: 300,
      width: 500,
    },
    series: [{
      name: 'Outdoor Temperature °C',
      data: {{contextos.outdoor|safe}}
    }],
    xaxis: {
      categories: {{contextos.data|safe}}
    },
    title: {
      text: 'Outdoor Temperature °C',
      align: 'left'
    },
  }

  //third chart
  var umidityOptions = {
    chart: {
      type: 'area',
      height: 300,
      width: 500,
    },
    series: [{
      name: 'Umidade relativa do ar',
      data: {{contextos.humidity|safe}}
    }],
    xaxis: {
      categories: {{contextos.data|safe}}
    },
    title: {
      text: 'Umidade relativa',
      align: 'left'
    },
  }

  //fourth chart
  var pressureOptions = {
    chart: {
      type: 'area',
      height: 300,
      width: 500,
    },
    series: [{
      name: 'Pressão atmosférica',
      data: {{contextos.air_pressure|safe}}
    }],
    xaxis: {
      categories: {{contextos.data|safe}}
    },
    title: {
      text: 'Pressão atmosférica',
      align: 'left'
    },
  }

  var temperaturaChart = new ApexCharts(document.querySelector("#temperatura"), tempOptions);
  temperaturaChart.render();

  var rainChart = new ApexCharts(document.querySelector("#milimetros"), rainOptions);
  rainChart.render();

  var umidityChart = new ApexCharts(document.querySelector("#umidade"), umidityOptions);
  umidityChart.render();

  var pressureChart = new ApexCharts(document.querySelector("#pressao"), pressureOptions);
  pressureChart.render();
};