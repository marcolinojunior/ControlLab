<template>
  <div class="input-control-panel">
    <h2>Analisador de Sistemas de Controle</h2>
    <p>Digite uma função de transferência ou um comando para iniciar a análise.</p>
    <div class="input-area">
      <textarea
        v-model="userInput"
        class="main-input"
        placeholder="Ex: G(s) = 1/(s^2 + 2*s + 1)"
        @keyup.enter="submitQuery"
      ></textarea>
      <button @click="submitQuery" class="analyze-button" :disabled="isLoading">
        {{ isLoading ? 'Analisando...' : 'Analisar' }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const userInput = ref('');
const isLoading = ref(false);

const emit = defineEmits(['query-submitted']);

const submitQuery = () => {
  if (userInput.value.trim()) {
    emit('query-submitted', userInput.value);
  }
};

// Method to update loading state, can be called by parent
const setLoading = (loading: boolean) => {
  isLoading.value = loading;
};

// Expose the setLoading method to the parent component
defineExpose({ setLoading });
</script>

<style scoped>
.input-control-panel {
  background-color: #ffffff;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.07);
}

h2 {
  margin-top: 0;
  margin-bottom: 0.5rem;
  color: #343a40;
}

p {
  margin-bottom: 1.5rem;
  color: #6c757d;
}

.input-area {
  display: flex;
  gap: 1rem;
}

.main-input {
  flex-grow: 1;
  padding: 0.75rem 1rem;
  font-family: 'Courier New', Courier, monospace;
  font-size: 1rem;
  border: 1px solid #ced4da;
  border-radius: 8px;
  resize: vertical;
  min-height: 50px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.main-input:focus {
  outline: none;
  border-color: #3a86ff; /* Azure */
  box-shadow: 0 0 0 3px rgba(58, 134, 255, 0.2);
}

.analyze-button {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: 600;
  color: #fff;
  background-color: #3a86ff; /* Azure */
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
}

.analyze-button:hover {
  background-color: #005ff9; /* Darker Azure */
  transform: translateY(-1px);
}

.analyze-button:disabled {
  background-color: #6c757d;
  cursor: not-allowed;
  transform: translateY(0);
}
</style>
