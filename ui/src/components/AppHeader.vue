<script setup lang="ts">
defineProps<{ runName: string; statusText: string }>()
</script>

<template>
  <header>
    <span class="brand" title="JEPA-RL: Joint-Embedding Predictive Architecture for Reinforcement Learning">jepa-rl</span>
    <span class="sep"></span>
    <span class="run-name" title="Current active run directory">{{ runName }}</span>
    <span class="header-status">{{ statusText }}</span>
  </header>
</template>
