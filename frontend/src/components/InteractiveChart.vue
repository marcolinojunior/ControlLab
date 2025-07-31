<template>
  <div class="chart-panel">
    <h3>Visualização Gráfica</h3>
    <div v-if="!chartOptions" class="no-chart">
      <p>O gráfico será exibido aqui quando os dados estiverem disponíveis.</p>
    </div>
    <div v-else>
      <highcharts :options="chartOptions" ref="chartRef"></highcharts>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { Chart as Highcharts } from 'highcharts-vue';
import HighchartsDarkUnica from 'highcharts/themes/dark-unica';
import HighchartsMore from 'highcharts/highcharts-more';

// Initialize Highcharts modules
if (typeof Highcharts === 'object') {
    HighchartsDarkUnica(Highcharts as any);
    HighchartsMore(Highcharts as any);
}

const props = defineProps<{
  plotData: any; // The plotData object from the backend response
}>();

const chartRef = ref<InstanceType<typeof Highcharts> | null>(null);
const chartOptions = ref<any>(null);

const baseChartOptions = {
  chart: {
    backgroundColor: '#1e1e1e'
  },
  credits: { enabled: false },
  legend: {
    itemStyle: {
        color: '#f0f0f0'
    },
    itemHoverStyle: {
        color: '#ffbe0b'
    }
  },
  plotOptions: {
    series: {
      color: '#ff006e', // Rosa (cor primária)
      marker: {
        enabled: false
      }
    }
  }
};

watch(() => props.plotData, (newData) => {
  if (!newData || !newData.type) {
    chartOptions.value = null;
    return;
  }

  let options = {};
  switch (newData.type) {
    case 'step_response':
      options = {
        title: { text: newData.title },
        xAxis: { title: { text: newData.xlabel } },
        yAxis: { title: { text: newData.ylabel } },
        series: [{
          name: 'Amplitude',
          data: newData.y.map((y_val: number, i: number) => [newData.x[i], y_val])
        }]
      };
      break;

    case 'bode':
      options = {
        title: { text: newData.title },
        xAxis: { type: 'logarithmic', title: { text: newData.xlabel } },
        yAxis: [
          { title: { text: newData.ylabel1 }, lineColor: '#ff006e' },
          { title: { text: newData.ylabel2 }, opposite: true, lineColor: '#fb5607' }
        ],
        series: [
          { name: 'Magnitude', yAxis: 0, data: newData.y1.map((y: number, i: number) => [newData.x[i], y]), color: '#ff006e' },
          { name: 'Fase', yAxis: 1, data: newData.y2.map((y: number, i: number) => [newData.x[i], y]), color: '#fb5607' }
        ]
      };
      break;

    case 'root_locus':
        const seriesData = newData.real.map((real: number, index: number) => [real, newData.imag[index]]);
        options = {
            chart: { type: 'scatter' },
            title: { text: newData.title },
            xAxis: { title: { text: newData.xlabel }, gridLineWidth: 1 },
            yAxis: { title: { text: newData.ylabel } },
            series: [{
                name: 'Lugar das Raízes',
                data: seriesData
            }]
        };
        break;
  }

  chartOptions.value = { ...baseChartOptions, ...options };

}, { deep: true });
</script>

<style scoped>
.chart-panel {
  background-color: #242424; /* Match workspace background */
  padding: 2rem;
  border-radius: 12px;
  min-height: 400px;
}

h3 {
  margin-top: 0;
  border-bottom: 1px solid #444;
  padding-bottom: 1rem;
  margin-bottom: 1rem;
  color: #fff;
}

.no-chart {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  text-align: center;
  color: #6c757d;
  padding: 3rem;
  border: 2px dashed #444;
  border-radius: 8px;
}
</style>
