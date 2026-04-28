import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { getJson } from '../api/client'

export interface Job {
  status: string
  run_name: string
  run_dir: string
  requested_steps: number
  error: string | null
  started_at: number
  completed_at: number | null
  running: boolean
}

export interface EvalJob {
  status: string
  run_name: string
  running: boolean
  episode_count: number
  episodes_target: number
  result: { mean_score?: number; best_score?: number } | null
  error: string | null
}

export const useTrainingStore = defineStore('training', () => {
  const summary = ref<Record<string, unknown>>({})
  const latestStep = ref<Record<string, unknown>>({})
  const steps = ref<Record<string, unknown>[]>([])
  const episodes = ref<Record<string, unknown>[]>([])
  const job = ref<Job | null>(null)
  const evalJob = ref<EvalJob | null>(null)
  const gameSettings = ref<[string, unknown][]>([])
  const configDetail = ref<ConfigGroup[]>([])
  const runDir = ref<{ name: string; dir: string; has_checkpoint: boolean; checkpoints: string[] } | null>(null)
  const resetKey = ref('Space')

  const isTraining = computed(() => !!job.value?.running || job.value?.status === 'running' || job.value?.status === 'starting')
  const isEvaluating = computed(() => !isTraining.value && (!!evalJob.value?.running || evalJob.value?.status === 'running'))
  const headerStatus = computed(() => job.value?.error ? `error: ${job.value.error}` : '')

  function chartPoints(field: string): [number, number][] {
    const key = field === 'td' ? 'td_error' : field
    return steps.value
      .filter(e => key === 'score' || key === 'epsilon' || e[key] != null)
      .map(e => [e.step as number, e[key] as number])
  }

  async function refresh() {
    try {
      const data = await getJson<Record<string, unknown>>('/api/state')
      const s = data.summary as Record<string, unknown> || {}
      const j = data.job as Job | null
      const e = data.eval as EvalJob | null
      const ls = data.latest_step as Record<string, unknown> || {}

      summary.value = s
      latestStep.value = ls
      steps.value = (data.steps as Record<string, unknown>[]) || []
      episodes.value = (data.episodes as Record<string, unknown>[]) || []
      job.value = j
      evalJob.value = e
      gameSettings.value = (data.game_settings as [string, unknown][]) || []
      configDetail.value = (data.config_detail as ConfigGroup[]) || []
      runDir.value = data.run as { name: string; dir: string; has_checkpoint: boolean; checkpoints: string[] } | null
      const cfg = data.config as { reset_key?: string } | undefined
      resetKey.value = cfg?.reset_key || 'Space'
    } catch { /* swallow */ }
  }

  const evalResult = computed(() => evalJob.value?.result ?? null)

  return { summary, latestStep, steps, episodes, job, evalJob, evalResult, gameSettings, configDetail, runDir, resetKey, isTraining, isEvaluating, headerStatus, chartPoints, refresh }
})

export interface ConfigGroup {
  title: string
  collapsed?: boolean
  fields: [string, unknown, string, { type: string; options?: string[]; min?: number; max?: number; step?: number }][]
}
