<script setup lang="ts">
import { RouterView } from 'vue-router'
</script>

<template>
  <header class="app-header">
    <div class="logo">
      <h1>ControlLab</h1>
      <span class="subtitle">Next-Gen Control Systems Analysis</span>
    </div>
  </header>

  <main>
    <RouterView />
  </main>
</template>

<style>
/* Global Styles */
body {
  background-color: #f0f2f5;
  color: #333;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
</style>

<style scoped>
.app-header {
  background-color: #ffffff;
  padding: 1rem 2rem;
  border-bottom: 1px solid #e6d7fb; /* Light Blue Violet */
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.logo h1 {
  margin: 0;
  font-size: 1.75rem;
  font-weight: 600;
  color: #8338ec; /* Blue Violet */
}

.logo .subtitle {
  font-size: 0.9rem;
  color: #8338ec; /* Blue Violet */
  opacity: 0.7;
}

main {
  padding: 2rem;
}
</style>
