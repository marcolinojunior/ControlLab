<template>
  <div class="home-view">
    <InputControl ref="inputControlRef" @query-submitted="handleQuery" />
    <div v-if="error" class="error-message">
      <strong>Erro na comunicação:</strong> {{ error }}
    </div>
    <div v-if="analysisResult" class="results-grid">
      <SolutionDisplay :analysis="analysisResult" />
      <InteractiveChart :plotData="analysisResult?.plotData" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import InputControl from '@/components/InputControl.vue';
import SolutionDisplay from '@/components/SolutionDisplay.vue';
import InteractiveChart from '@/components/InteractiveChart.vue';

const inputControlRef = ref<InstanceType<typeof InputControl> | null>(null);
const analysisResult = ref<any>(null);
const error = ref<string | null>(null);

const orchestratorUrl = '/api/query'; // Using relative URL for proxy

const handleQuery = async (userInput: string) => {
  if (inputControlRef.value) {
    inputControlRef.value.setLoading(true);
  }
  error.value = null;
  analysisResult.value = null;

  try {
    const response = await axios.post(orchestratorUrl, { query: userInput });
    analysisResult.value = response.data;
  } catch (err: any) {
    if (err.response) {
      error.value = `Erro do servidor (${err.response.status}): ${err.response.data.error || err.message}`;
    } else if (err.request) {
      error.value = 'Não foi possível conectar ao servidor do orquestrador. Ele está em execução?';
    } else {
      error.value = `Erro inesperado: ${err.message}`;
    }
  } finally {
    if (inputControlRef.value) {
      inputControlRef.value.setLoading(false);
    }
  }
};
</script>

<style scoped>
.home-view {
  max-width: 1400px;
  margin: 0 auto;
}

.error-message {
  margin-top: 1rem;
  color: #721c24;
  background-color: #f8d7da;
  border: 1px solid #f5c6cb;
  padding: 1rem;
  border-radius: 4px;
}

.results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

@media (max-width: 992px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
}
</style>
