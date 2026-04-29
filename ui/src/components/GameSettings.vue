<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  settings: [string, unknown][]
}>()

const description = computed(() => {
  const entry = props.settings.find(([k]) => k === 'description')
  return entry ? String(entry[1]) : ''
})
const filteredSettings = computed(() => props.settings.filter(([k]) => k !== 'description'))
</script>

<template>
  <div class="settings-table" title="Active game controls">
    <template v-if="description">
      <span class="sk">description</span>
      <span class="sv game-desc">{{ description }}</span>
    </template>
    <template v-for="[key, val] in filteredSettings" :key="key">
      <span class="sk">{{ key }}</span>
      <span class="sv">{{ String(val) }}</span>
    </template>
  </div>
</template>
