<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  summary: Record<string, unknown>
  latestStep: Record<string, unknown>
  evalResult: { mean_score?: number } | null
}>()

const items = computed(() => {
  const s = props.summary || {}
  const l = props.latestStep || {}
  const list: [string, unknown][] = [
    ['steps', s.steps ?? l.step],
    ['episodes', s.episodes],
    ['best score', s.best_score],
    ['updates', l.updates ?? s.update_count],
    ['replay', l.replay_size ?? s.replay_size],
    ['weight delta', l.weight_delta_norm ?? s.weight_delta_norm],
    ['target syncs', l.target_updates ?? s.target_update_count],
  ]
  if (props.evalResult?.mean_score != null) {
    list.push(['eval mean', props.evalResult.mean_score])
  }
  return list
})

function fmt(v: unknown): string {
  if (v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v))) return '—'
  if (typeof v === 'number') {
    if (Math.abs(v) >= 1000) return v.toFixed(0)
    if (Math.abs(v) >= 10) return v.toFixed(2)
    return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  }
  return String(v)
}
</script>

<template>
  <div class="metrics" title="Live training metrics updated every refresh">
    <div v-for="[label, value] in items" :key="label" class="metric">
      <div class="m-label">{{ label }}</div>
      <div class="m-val">{{ fmt(value) }}</div>
    </div>
  </div>
</template>
