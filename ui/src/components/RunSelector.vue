<script setup lang="ts">
import { ref } from 'vue'
import { useRunsStore } from '../stores/runs'

const runs = useRunsStore()
const showNewModal = ref(false)
const newRunName = ref('')

function onChange() {
  runs.loadRunDetail(runs.selectedRun)
}

function openNewModal() {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  newRunName.value = `run_${ts}`
  showNewModal.value = true
}

function cancelNew() {
  showNewModal.value = false
  newRunName.value = ''
}

function confirmNew() {
  const name = newRunName.value.trim()
  if (!name) return
  runs.selectedRun = ''
  runs.checkpoints = []
  runs.runDir = ''
  runs.runConfigDetail = null
  runs.$patch && runs.$patch({})
  // Dispatch custom event so TrainControls can pick up the new name
  window.dispatchEvent(new CustomEvent('new-run', { detail: { name } }))
  showNewModal.value = false
}

async function deleteSelectedRun() {
  if (!runs.selectedRun) return
  if (!confirm(`Delete run "${runs.selectedRun}" and all its data?`)) return
  try {
    await runs.deleteRun(runs.selectedRun)
  } catch (e) {
    alert(e instanceof Error ? e.message : String(e))
  }
}

function fmt(v: unknown): string {
  if (v == null || (typeof v === 'number' && Number.isNaN(v))) return '—'
  if (typeof v === 'number') {
    if (Math.abs(v) >= 1000) return v.toFixed(0)
    if (Math.abs(v) >= 10) return v.toFixed(2)
    return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  }
  return String(v)
}

function runLabel(r: { name: string; algorithm?: string; best_score?: number; steps?: number }): string {
  const parts = [r.name]
  if (r.algorithm) parts.push(r.algorithm)
  if (r.best_score != null) parts.push('best:' + fmt(r.best_score))
  if (r.steps != null) parts.push(r.steps + ' steps')
  return parts.join(' · ')
}
</script>

<template>
  <div class="run-select">
    <select v-model="runs.selectedRun" @change="onChange" title="Select a past training run to view its details">
      <option value="">select run...</option>
      <option v-for="r in runs.runs" :key="r.name" :value="r.name">{{ runLabel(r) }}</option>
    </select>
    <div class="run-select-actions">
      <button @click="openNewModal" class="btn-tiny" title="Start a new training run">+ new run</button>
      <label class="smoke-toggle">
        <input type="checkbox" v-model="runs.showSmoke" @change="runs.loadRuns()"> show smoke
      </label>
      <button v-if="runs.selectedRun" @click="deleteSelectedRun" class="btn-danger-tiny">delete</button>
    </div>
    <!-- inline modal for new run name -->
    <div v-if="showNewModal" class="modal-overlay" @click.self="cancelNew">
      <div class="modal-box">
        <div class="modal-title">new run</div>
        <input
          v-model="newRunName"
          class="modal-input"
          placeholder="run name"
          @keydown.enter="confirmNew"
          @keydown.escape="cancelNew"
          autofocus
        />
        <div class="modal-actions">
          <button @click="cancelNew" class="btn-tiny">cancel</button>
          <button @click="confirmNew" class="btn-accent-tiny">create</button>
        </div>
      </div>
    </div>
  </div>
</template>
