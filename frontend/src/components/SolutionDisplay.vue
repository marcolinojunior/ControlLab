<template>
  <div class="solution-display-panel" v-if="analysis">
    <h3>Resultado da Análise</h3>
    <div v-if="analysis.success === false" class="error-message">
      <strong>Erro:</strong> {{ analysis.message || 'Ocorreu um erro na análise.' }}
    </div>
    <div v-else class="steps-container">
      <div v-for="(step, index) in formattedSteps" :key="index" class="analysis-step">
        <p v-html="step.text"></p>
        <div v-if="step.latex" v-html="renderLatex(step.latex)" class="math-expression"></div>
      </div>
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

const formattedSteps = computed(() => {
  if (!props.analysis?.analysis?.steps) {
    // Handle cases where the structure might be different, e.g. direct stability analysis
    if (props.analysis?.analysis?.stability_analysis) {
        return [{ text: JSON.stringify(props.analysis.analysis.stability_analysis, null, 2) }];
    }
    return [];
  }
  return props.analysis.analysis.steps;
});

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

.error-message {
  color: #fff;
  background-color: #ff006e; /* Rose */
  border: 1px solid #cc0058; /* Darker Rose */
  padding: 1rem;
  border-radius: 8px;
  font-weight: 500;
}

.analysis-step {
  margin-bottom: 1.5rem;
  padding: 1rem;
  border-left: 4px solid #ffbe0b; /* Amber */
  background-color: #fff;
}

.math-expression {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  text-align: center;
  margin-top: 1rem;
}

.katex-error {
  color: #ff006e; /* Rose */
  font-family: monospace;
}
</style>
