<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useRunsStore } from '../stores/runs'
import { api, getJson } from '../api/client'

const training = useTrainingStore()
const runs = useRunsStore()

const runName = ref('ui_train')
const steps = ref(500)
const warmup = ref(50)
const batch = ref('')
const logEvery = ref(5)
const selectedCheckpoint = ref('')
const controlError = ref('')

const busy = computed(() => training.isTraining || training.isEvaluating)
const availableCheckpoints = computed(() => {
  if (runs.checkpoints.length) return runs.checkpoints
  return training.runDir?.checkpoints ?? []
})
const evalRunDir = computed(() => runs.runDir || training.runDir?.dir || '')
const hasCheckpoint = computed(() => availableCheckpoints.value.length > 0)

watch(availableCheckpoints, checkpoints => {
  if (!checkpoints.length) {
    selectedCheckpoint.value = ''
    return
  }
  if (!selectedCheckpoint.value || !checkpoints.includes(selectedCheckpoint.value)) {
    selectedCheckpoint.value = checkpoints[0]
  }
}, { immediate: true })

onMounted(async () => {
  try {
    const defaults = await getJson<{
      run_name: string
      default_steps: number
      learning_starts: number
      batch_size: number | null
      dashboard_every: number
    }>('/api/defaults')
    runName.value = defaults.run_name || 'ui_train'
    steps.value = defaults.default_steps || 500
    warmup.value = defaults.learning_starts ?? 50
    batch.value = defaults.batch_size != null ? String(defaults.batch_size) : ''
    logEvery.value = defaults.dashboard_every || 5
  } catch { /* use defaults */ }
})

async function startTraining() {
  controlError.value = ''
  try {
    await api('/api/train/start', {
      experiment: runName.value,
      steps: Number(steps.value),
      learning_starts: Number(warmup.value),
      batch_size: batch.value || null,
      dashboard_every: Number(logEvery.value) || 5,
    })
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
  setTimeout(() => runs.loadRuns(), 500)
}

async function stopTraining() {
  controlError.value = ''
  try {
    if (training.isEvaluating) {
      await api('/api/eval/stop')
    } else {
      await api('/api/train/stop')
    }
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
}

async function watchAiPlay() {
  controlError.value = ''
  if (!selectedCheckpoint.value) {
    controlError.value = 'select a checkpoint first'
    return
  }
  const payload: Record<string, unknown> = { episodes: 3, checkpoint: selectedCheckpoint.value }
  if (evalRunDir.value) payload.run_dir = evalRunDir.value
  try {
    await api('/api/eval/start', payload)
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
}
</script>

<template>
  <div class="train-controls" id="controlGroup" :data-run-dir="runs.runDir">
    <div class="tc-row">
      <div class="field" title="Name for the new training run">name <input v-model="runName" type="text" style="width:80px" title="Run name for new training"></div>
      <div class="field" title="Total environment steps">steps <input v-model.number="steps" type="number" min="1" title="Number of game steps to train"></div>
      <div class="field" title="Steps before model updates begin">warmup <input v-model.number="warmup" type="number" min="0" title="Warmup steps"></div>
    </div>
    <div class="tc-row">
      <div class="field" title="Minibatch size">batch <input v-model="batch" type="number" min="1" placeholder="auto" title="Batch size override"></div>
      <div class="field" title="Rewrite dashboard every N steps">log&nbsp;every <input v-model.number="logEvery" type="number" min="1" title="Dashboard write interval"></div>
    </div>
    <div class="tc-row cg-ckpt" v-if="hasCheckpoint">
      <span class="cg-ckpt-label">checkpoint</span>
      <select v-model="selectedCheckpoint" title="Available model checkpoints">
        <option v-for="c in availableCheckpoints" :key="c" :value="c">{{ c }}</option>
      </select>
    </div>
    <div class="tc-row tc-error" v-if="controlError">{{ controlError }}</div>
    <div class="tc-row tc-actions">
      <button @click="startTraining" v-if="!busy" class="btn-accent" title="Start training">train</button>
      <button @click="watchAiPlay" v-if="!busy && hasCheckpoint" class="btn-accent" title="Watch AI play">watch AI play</button>
      <button @click="stopTraining" v-if="busy" class="btn-danger" title="Stop training/eval">stop</button>
    </div>
  </div>
</template>
