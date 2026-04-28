<script setup lang="ts">
import { computed } from 'vue'
import type { Job, EvalJob } from '../stores/training'

const props = defineProps<{
  job: Job | null
  evalJob: EvalJob | null
  summary: Record<string, unknown>
  latestStep: Record<string, unknown>
}>()

const isTraining = computed(() => !!props.job?.running || props.job?.status === 'running' || props.job?.status === 'starting')
const isEvaluating = computed(() => !isTraining.value && (!!props.evalJob?.running || props.evalJob?.status === 'running'))

const dotClass = computed(() => {
  if (isTraining.value) return 'rib-dot running'
  if (isEvaluating.value) return 'rib-dot evaluating'
  if (props.job?.status === 'error' || props.evalJob?.status === 'error') return 'rib-dot error'
  if (props.job?.status === 'completed') return 'rib-dot stopped'
  return 'rib-dot idle'
})

const statusText = computed(() => {
  if (isTraining.value) return 'training'
  if (isEvaluating.value) return 'evaluating'
  if (props.job?.status === 'completed') return 'completed'
  if (props.job?.status === 'stopped') return 'stopped'
  if (props.job?.status === 'error') return 'error'
  if (props.evalJob?.status === 'error') return 'eval error'
  return 'idle'
})

const detail = computed(() => {
  const s = props.summary || {}
  const parts: string[] = []
  if (s.algorithm) parts.push(String(s.algorithm))
  if (s.episodes) parts.push(`${s.episodes} eps`)
  if (s.best_score != null) parts.push('best:' + fmtNum(s.best_score))
  if (s.steps) parts.push(fmtNum(s.steps) + ' steps')
  return parts.join(' · ')
})

function fmtNum(v: unknown): string {
  if (v == null || typeof v !== 'number') return '—'
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}
</script>

<template>
  <div class="run-info-bar" title="Current run status and key metrics">
    <span :class="dotClass" id="ribDot"></span>
    <span class="rib-status">{{ statusText }}</span>
    <span class="rib-detail">{{ detail }}</span>
  </div>
</template>
