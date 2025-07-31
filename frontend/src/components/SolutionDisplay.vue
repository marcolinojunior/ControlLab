<template>
  <div class="solution-display-panel" v-if="analysis">
    <h3>Solução Detalhada</h3>
    <div v-if="analysis.error" class="error-message">
      <strong>Erro:</strong> {{ analysis.error.message || 'Ocorreu um erro na análise.' }}
    </div>
    <div v-else-if="analysis.solution" class="solution-container">
        <p v-if="analysis.solution.text">{{ analysis.solution.text }}</p>
        <div v-if="analysis.solution.latex" v-html="renderLatex(analysis.solution.latex)" class="math-expression"></div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const props = defineProps<{
  analysis: any; // The full response object from the orchestrator
}>();

const renderLatex = (latexString: string) => {
  if (!latexString) return '';
  try {
    return katex.renderToString(latexString, {
      throwOnError: false,
      displayMode: true,
    });
  } catch (e) {
    console.error('KaTeX rendering error:', e);
    return `<span class="katex-error">${latexString}</span>`;
  }
};
</script>

<style scoped>
.solution-display-panel {
  background-color: #242424; /* Match workspace background */
  padding: 2rem;
  border-radius: 12px;
}

h3 {
  margin-top: 0;
  border-bottom: 1px solid #444;
  padding-bottom: 1rem;
  margin-bottom: 1rem;
  color: #fff;
}

.error-message {
  color: #fff;
  background-color: #ff006e; /* Rosa */
  border: 1px solid #cc0058;
  padding: 1rem;
  border-radius: 8px;
  font-weight: 500;
}

.solution-container p {
  color: #f0f0f0;
  font-size: 1.1rem;
  line-height: 1.6;
}

.math-expression {
  background: #1e1e1e; /* Quase preto, para destaque */
  padding: 1.5rem;
  border-radius: 8px;
  overflow-x: auto;
  text-align: center;
  margin-top: 1rem;
  color: #3a86ff; /* Azul Azure para as fórmulas */
}

.katex-error {
  color: #ff006e; /* Rosa */
  font-family: monospace;
}
</style>
