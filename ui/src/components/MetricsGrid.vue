<script setup lang="ts">
import { computed } from 'vue'
import type { Job, CollectJob } from '../stores/training'

const props = defineProps<{
  summary: Record<string, unknown>
  latestStep: Record<string, unknown>
  evalResult: { mean_score?: number } | null
  worldJob: Job | null
  collectJob: CollectJob | null
}>()

const isWorldActive = computed(() => {
  const wj = props.worldJob
  return wj && (wj.running || wj.status === 'running' || wj.status === 'starting' || wj.status === 'completed' || wj.status === 'error')
})

const items = computed(() => {
  if (isWorldActive.value) return worldMetrics.value
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

const worldMetrics = computed(() => {
  const wj = props.worldJob
  const ls = props.latestStep || {}
  const list: [string, unknown][] = []

  if (wj?.status === 'completed') {
    list.push(['status', 'done'])
  } else if (wj?.status === 'error') {
    list.push(['status', 'error'])
  } else if (ls.phase === 'collecting') {
    list.push(['collected', `${ls.collect_step ?? 0}/${ls.collect_total ?? '?'}`])
    list.push(['episodes', ls.episodes ?? 0])
  } else if (typeof ls.step === 'number' && (ls.phase !== 'collecting')) {
    list.push(['step', `${ls.step}/${wj?.requested_steps ?? 0}`])
  }

  if (typeof ls.loss === 'number') list.push(['loss', ls.loss])
  if (typeof ls.prediction_loss === 'number') list.push(['pred loss', ls.prediction_loss])
  if (typeof ls.variance_loss === 'number') list.push(['var loss', ls.variance_loss])
  if (typeof ls.tau === 'number') list.push(['EMA tau', ls.tau])
  if (typeof ls.latent_std_mean === 'number') list.push(['latent std', ls.latent_std_mean])
  if (typeof ls.effective_rank === 'number') list.push(['eff rank', ls.effective_rank])
  if (typeof ls.replay_size === 'number') list.push(['replay', ls.replay_size])

  return list
})

const collectItems = computed(() => {
  const cj = props.collectJob
  if (!cj || (!cj.running && cj.status !== 'completed')) return []
  return [
    ['collect eps', `${cj.episodes_done}/${cj.episodes_target}`],
    ['mean score', cj.mean_score],
  ] as [string, unknown][]
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
    <template v-if="collectItems.length">
      <div class="metric metric-divider" v-for="[label, value] in collectItems" :key="'c-' + label">
        <div class="m-label">{{ label }}</div>
        <div class="m-val">{{ fmt(value) }}</div>
      </div>
    </template>
  </div>
</template>
