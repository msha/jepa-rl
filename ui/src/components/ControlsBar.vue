<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useConfigStore } from '../stores/config'
import { useTrainingStore } from '../stores/training'

const config = useConfigStore()
const training = useTrainingStore()

const selectedConfig = ref('')
const validateStatus = ref<'idle' | 'ok' | 'error'>('idle')
const validateMsg = ref('')

onMounted(async () => {
  await config.loadConfigs()
  selectedConfig.value = config.currentPath
})

watch(() => config.currentPath, (p) => { selectedConfig.value = p })

async function onConfigChange() {
  if (!selectedConfig.value) return
  validateStatus.value = 'idle'
  validateMsg.value = ''
  try {
    await config.switchConfig(selectedConfig.value)
    const result = await training.validateConfig(selectedConfig.value)
    if (result.ok) {
      validateStatus.value = 'ok'
      validateMsg.value = `${result.game} / ${result.algorithm}`
    } else {
      validateStatus.value = 'error'
      validateMsg.value = result.error || 'invalid config'
    }
    await training.refresh()
  } catch (e) {
    validateStatus.value = 'error'
    validateMsg.value = e instanceof Error ? e.message : String(e)
  }
}
</script>

<template>
  <div class="controls-bar">
    <div class="field">
      config
      <select v-model="selectedConfig" @change="onConfigChange" title="Switch active game config">
        <option v-for="c in config.configs" :key="c.path" :value="c.path">{{ c.name }}</option>
      </select>
    </div>
    <span v-if="validateStatus === 'ok'" class="cb-validate ok" :title="validateMsg">✓ {{ validateMsg }}</span>
    <span v-if="validateStatus === 'error'" class="cb-validate err" :title="validateMsg">✗ {{ validateMsg }}</span>
  </div>
</template>
