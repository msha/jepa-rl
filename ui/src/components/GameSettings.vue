<script setup lang="ts">
defineProps<{ settings: [string, unknown][] }>()
</script>

<template>
  <div class="settings-table" title="Game configuration settings for the active config">
    <template v-for="[key, val] in settings" :key="key">
      <span class="sk">{{ key }}</span>
      <span class="sv">{{ String(val) }}</span>
    </template>
  </div>
</template>
