<template>
  <div class="action-buttons-panel">
    <h4>Ações para o Sistema '{{ systemName }}':</h4>
    <div class="buttons-wrapper">
      <button @click="emitCommand('step_response')">Resposta ao Degrau</button>
      <button @click="emitCommand('bode_plot')">Diagrama de Bode</button>
      <button @click="emitCommand('root_locus')">Lugar das Raízes</button>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps({
  systemName: {
    type: String,
    required: true
  }
});

const emit = defineEmits(['command']);

function emitCommand(analysisType: string) {
  // Emits the command in the pipeline format, applied to the current system
  emit('command', `${props.systemName} | ${analysisType}`);
}
</script>

<style scoped>
.action-buttons-panel {
  margin-top: 1.5rem;
  padding: 1.5rem;
  background-color: #242424;
  border-radius: 12px;
  border: 1px solid #3a86ff; /* Azul Azure */
}

h4 {
  margin-top: 0;
  margin-bottom: 1rem;
  color: #fff;
  font-weight: 500;
}

.buttons-wrapper {
  display: flex;
  gap: 0.75rem;
}

.buttons-wrapper button {
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: #fff;
  background-color: #3a86ff; /* Azul Azure */
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.buttons-wrapper button:hover {
  background-color: #005ff9;
}
</style>
