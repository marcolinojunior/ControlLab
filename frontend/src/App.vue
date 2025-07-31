<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <div class="logo">
        <h1>Controlab</h1>
      </div>
      <nav class="navigation">
        <a href="#">Documentação</a>
        <a href="#">Exemplos</a>
        <a href="#">Sobre</a>
      </nav>
    </header>

    <main class="main-workspace">
      <RouterView />
    </main>

    <footer class="app-footer">
      <p>© 2025 Controlab. Todos os direitos reservados.</p>
    </footer>
  </div>
</template>

<style>
/* Global styles moved to main.css, this block can be for unscoped overrides if needed */
</style>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

.app-header {
  background-color: #1a1a1a;
  padding: 1rem 2rem;
  border-bottom: 1px solid #333;
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  box-sizing: border-box;
}

.logo h1 {
  margin: 0;
  font-size: 1.75rem;
  font-weight: 600;
  color: #3a86ff; /* Azul Azure */
}

.navigation a {
  color: #f0f0f0;
  margin-left: 1.5rem;
  font-weight: 500;
}

.navigation a:hover {
  color: #ffbe0b; /* Amarelo Âmbar */
}

.main-workspace {
  flex-grow: 1;
  background-color: #242424; /* Cinza escuro para a área de trabalho */
  padding: 2rem;
}

.app-footer {
  background-color: #1a1a1a;
  padding: 1.5rem 2rem;
  border-top: 1px solid #333;
  text-align: center;
  color: #888;
  font-size: 0.9rem;
}
</style>
