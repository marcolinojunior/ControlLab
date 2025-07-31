<template>
  <div class="chart-panel">
    <h3>Visualizações Gráficas</h3>
    <div v-if="chartOptions">
      <highcharts :options="chartOptions"></highcharts>
    </div>
    <div v-else class="no-chart">
      <p>O gráfico será exibido aqui quando os dados estiverem disponíveis.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { Chart } from 'highcharts-vue';

const props = defineProps<{
  plotData: any; // The plotData object from the backend response
}>();

const chartOptions = ref(null);

watch(() => props.plotData, (newData) => {
  if (newData && newData.type) {
    // Basic example: Handle a generic 'root-locus' type
    if (newData.type === 'root-locus') {
      chartOptions.value = {
        chart: {
          type: 'scatter',
        },
        title: {
          text: 'Lugar das Raízes (Root Locus)',
        },
        xAxis: {
          title: {
            text: 'Real',
          },
          gridLineWidth: 1,
        },
        yAxis: {
          title: {
            text: 'Imaginário',
          },
        },
        series: [
          ...newData.branches.map((branch, index) => ({
            name: `Branch ${index + 1}`,
            data: branch.x.map((val, i) => [val, branch.y[i]]),
          })),
          {
            type: 'scatter',
            name: 'Poles',
            data: newData.poles.map(p => [p.x, p.y]),
            marker: {
              symbol: 'cross',
              lineWidth: 2,
              radius: 8,
            },
            color: '#ff006e', /* Rose */
          },
          {
            type: 'scatter',
            name: 'Zeros',
            data: newData.zeros.map(z => [z.x, z.y]),
            marker: {
              symbol: 'circle',
              lineWidth: 2,
              radius: 8,
              fillColor: 'white',
            },
            color: '#8338ec', /* Blue Violet */
          }
        ],
        plotOptions: {
            series: {
                color: '#3a86ff' /* Azure */
            }
        }
      };
    }
    // Add handlers for other chart types like 'bode', 'step-response', etc. here
  } else {
    chartOptions.value = null;
  }
}, { deep: true });
</script>

<style scoped>
.chart-panel {
  background-color: #ffffff;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.07);
}

h3 {
  margin-top: 0;
  border-bottom: 1px solid #eee;
  padding-bottom: 1rem;
  margin-bottom: 1rem;
  color: #343a40;
}

.no-chart {
  text-align: center;
  color: #6c757d;
  padding: 3rem;
  border: 2px dashed #e9ecef;
  border-radius: 8px;
}
</style>
