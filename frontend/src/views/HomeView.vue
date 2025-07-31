<template>
  <div class="home-view">
    <InputControl @command="handleCommand" />
    <ActionButtons v-if="systemDefined" :systemName="systemName" @command="handleCommand" />

    <div v-if="isLoading" class="loading-spinner">Analisando...</div>

    <div v-if="error" class="error-message">
      <strong>Erro na comunicação:</strong> {{ error }}
    </div>

    <div v-if="analysisResult" class="results-grid">
      <SolutionDisplay :analysis="analysisResult" />
      <InteractiveChart :plotData="analysisResult?.chartData" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import InputControl from '@/components/InputControl.vue';
import ActionButtons from '@/components/ActionButtons.vue'; // Import the new component
import SolutionDisplay from '@/components/SolutionDisplay.vue';
import InteractiveChart from '@/components/InteractiveChart.vue';

const analysisResult = ref<any>(null);
const error = ref<string | null>(null);
const isLoading = ref(false);

// State for contextual buttons
const systemDefined = ref(false);
const systemName = ref('G'); // Default system name

const orchestratorUrl = '/api/control'; // Use the new endpoint

const handleCommand = async (command: string) => {
  isLoading.value = true;
  error.value = null;
  // Do not clear previous results, so the last chart stays visible
  // analysisResult.value = null;

  try {
    const response = await axios.post(orchestratorUrl, { command: command });
    analysisResult.value = response.data;

    // Logic to enable contextual buttons
    const commandDefinesSystem = command.includes('= tf(') || command.includes('= ss(');
    if (commandDefinesSystem) {
      systemDefined.value = true;
      // Extract the variable name from the command
      const match = command.match(/^\s*([a-zA-Z0-9_]+)\s*=/);
      if (match) {
        systemName.value = match[1];
      }
    }

  } catch (err: any) {
    if (err.response) {
      error.value = `Erro do servidor (${err.response.status}): ${err.response.data.error?.message || err.message}`;
    } else if (err.request) {
      error.value = 'Não foi possível conectar ao servidor do orquestrador. Ele está em execução?';
    } else {
      error.value = `Erro inesperado: ${err.message}`;
    }
  } finally {
    isLoading.value = false;
  }
};
</script>

<style scoped>
.home-view {
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
}

.loading-spinner {
  text-align: center;
  margin-top: 2rem;
  font-size: 1.2rem;
  color: #8338ec; /* Azul Violeta */
}

.error-message {
  margin-top: 1rem;
  color: #fb5607; /* Laranja Pantone */
  background-color: rgba(251, 86, 7, 0.1);
  border: 1px solid #fb5607;
  padding: 1rem;
  border-radius: 8px;
}

.results-grid {
  display: grid;
  grid-template-columns: 40% 60%;
  gap: 2rem;
  margin-top: 2rem;
}

@media (max-width: 992px) {
  .results-grid {
    grid-template-columns: 1fr;
  }
}
</style>
