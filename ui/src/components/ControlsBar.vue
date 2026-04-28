<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useConfigStore } from '../stores/config'

const config = useConfigStore()
const selected = ref('')

onMounted(async () => {
  await config.loadConfigs()
  selected.value = config.currentPath
})

async function onChange() {
  if (!selected.value) return
  try {
    await config.switchConfig(selected.value)
  } catch (e) {
    console.error(e)
  }
}
</script>

<template>
  <div class="controls-bar">
    <div class="field" title="Select a game configuration to load">
      game
      <select v-model="selected" @change="onChange" title="Available game configs in configs/games/">
        <option v-for="c in config.configs" :key="c.path" :value="c.path">{{ c.name }}</option>
      </select>
    </div>
  </div>
</template>
