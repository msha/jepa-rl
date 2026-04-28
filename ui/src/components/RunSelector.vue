<script setup lang="ts">
import { useRunsStore } from '../stores/runs'

const runs = useRunsStore()

function onChange() {
  if (runs.selectedRun) {
    runs.loadRunDetail(runs.selectedRun)
  } else {
    runs.checkpoints = []
    runs.runDir = ''
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
  </div>
</template>
