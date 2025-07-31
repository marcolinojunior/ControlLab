<template>
  <div>
    <div class="tool-panel">
      <button @click="addTransferFunction">Função de Transferência</button>
      <button @click="appendToCommand(' | step_response')">Resposta ao Degrau</button>
      <button @click="appendToCommand(' | bode_plot')">Diagrama de Bode</button>
      <button @click="appendToCommand(' | root_locus')">Lugar das Raízes</button>
    </div>

    <div class="input-area">
      <input v-model="command" @keyup.enter="sendCommand" placeholder="Construa seu comando aqui..." class="main-input" />
      <button @click="sendCommand" class="execute-button">Executar</button>
    </div>

    <div class="latex-preview" ref="latexPreview"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const command = ref('');
const latexPreview = ref<HTMLDivElement | null>(null);

const emit = defineEmits(['command']);

function appendToCommand(text: string) {
  // Add a space if the command is not empty and doesn't already have one
  if (command.value.length > 0 && !command.value.endsWith(' ')) {
    command.value += ' ';
  }
  command.value += text;
}

function addTransferFunction() {
  // Simulação: em um caso real, isso abriria um modal
  const num = prompt("Digite o numerador (ex: 1):", "1");
  const den = prompt("Digite o denominador (ex: 1,2,5):", "1,2,5");
  if (num && den) {
    // Overwrite the command with the new transfer function
    command.value = `G = tf([${num}], [${den}])`;
  }
}

function sendCommand() {
  emit('command', command.value);
}

// Assista a mudanças no comando e atualize a pré-visualização
watch(command, (newCommand) => {
  if (latexPreview.value) {
    try {
      // Tenta renderizar o comando inteiro. KaTeX vai ignorar o que não for matemática.
      // We'll wrap the non-pipe parts in LaTeX delimiters to render them.
      const parts = newCommand.split('|').map(part => part.trim());
      const latexString = parts.map(part => {
        if (part.includes('tf') || part.includes('ss')) {
          // Attempt to convert Python list syntax to matrix syntax for better rendering
          let processedPart = part.replace(/tf\(\[/g, 'tf([').replace(/\]\)/g, '])');
          processedPart = processedPart.replace(/\[/g, '{').replace(/\]/g, '}');
          return `$$${processedPart}$$`;
        }
        return part;
      }).join(' \\ | \\ ');

      katex.render(latexString, latexPreview.value, {
        throwOnError: false,
        displayMode: true
      });
    } catch (e) {
      if (e instanceof Error) {
        latexPreview.value.innerHTML = `<span style="color: #fb5607;">Erro de sintaxe KaTeX: ${e.message}</span>`;
      } else {
        latexPreview.value.innerHTML = `<span style="color: #fb5607;">Erro de sintaxe KaTeX</span>`;
      }
    }
  }
});

</script>

<style scoped>
.tool-panel {
  display: flex;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.tool-panel button {
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: #fff;
  background-color: #8338ec; /* Azul Violeta */
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.tool-panel button:hover {
  background-color: #6d1fe5;
}

.input-area {
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
}

.main-input {
  flex-grow: 1;
  padding: 0.75rem 1rem;
  font-family: 'Courier New', Courier, monospace;
  font-size: 1.1rem;
  color: #fff;
  background-color: #242424;
  border: 1px solid #444;
  border-radius: 8px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.main-input:focus {
  outline: none;
  border-color: #ffbe0b; /* Amarelo Âmbar */
  box-shadow: 0 0 0 3px rgba(255, 190, 11, 0.2);
}

.execute-button {
  padding: 0.75rem 2rem;
  font-size: 1.1rem;
  font-weight: 700;
  color: #000;
  background-color: #ffbe0b; /* Amarelo Âmbar */
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background-color 0.2s, transform 0.1s;
}

.execute-button:hover {
  background-color: #fca311;
  transform: translateY(-1px);
}

.latex-preview {
  margin-top: 1.5rem;
  padding: 1rem;
  min-height: 50px;
  background-color: #242424;
  border: 1px dashed #444;
  border-radius: 8px;
  color: #ffbe0b; /* Amarelo Âmbar */
  font-size: 1.2rem;
  text-align: center;
  overflow-x: auto;
}
</style>
