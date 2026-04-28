import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { api, getJson } from '../api/client'

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

export interface CollectJob {
  status: string
  run_name: string
  episodes_done: number
  episodes_target: number
  mean_score: number
  error: string | null
  started_at: number
  completed_at: number | null
  running: boolean
}

export const useTrainingStore = defineStore('training', () => {
  const summary = ref<Record<string, unknown>>({})
  const latestStep = ref<Record<string, unknown>>({})
  const steps = ref<Record<string, unknown>[]>([])
  const episodes = ref<Record<string, unknown>[]>([])
  const job = ref<Job | null>(null)
  const evalJob = ref<EvalJob | null>(null)
  const worldJob = ref<Job | null>(null)
  const collectJob = ref<CollectJob | null>(null)
  const gameSettings = ref<[string, unknown][]>([])
  const configDetail = ref<ConfigGroup[]>([])
  const runDir = ref<{ name: string; dir: string; has_checkpoint: boolean; checkpoints: { file: string; label: string }[] } | null>(null)
  const resetKey = ref('Space')
  const actionKeys = ref<string[]>([])

  const isTraining = computed(() => !!job.value?.running || job.value?.status === 'running' || job.value?.status === 'starting')
  const isEvaluating = computed(() => !isTraining.value && (!!evalJob.value?.running || evalJob.value?.status === 'running'))
  const isWorldTraining = computed(() => !!worldJob.value?.running || worldJob.value?.status === 'running' || worldJob.value?.status === 'starting')
  const isCollecting = computed(() => !!collectJob.value?.running || collectJob.value?.status === 'running' || collectJob.value?.status === 'starting')
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
      const wj = data.world_job as Job | null
      const cj = data.collect_job as CollectJob | null
      const ls = data.latest_step as Record<string, unknown> || {}

      summary.value = s
      latestStep.value = ls
      steps.value = (data.steps as Record<string, unknown>[]) || []
      episodes.value = (data.episodes as Record<string, unknown>[]) || []
      job.value = j
      evalJob.value = e
      worldJob.value = wj
      collectJob.value = cj
      gameSettings.value = (data.game_settings as [string, unknown][]) || []
      configDetail.value = (data.config_detail as ConfigGroup[]) || []
      runDir.value = data.run as { name: string; dir: string; has_checkpoint: boolean; checkpoints: { file: string; label: string }[] } | null
      const cfg = data.config as { reset_key?: string; action_keys?: string[] } | undefined
      resetKey.value = cfg?.reset_key || 'Space'
      actionKeys.value = cfg?.action_keys || []
    } catch { /* swallow */ }
  }

  async function validateConfig(configPath?: string): Promise<{ ok: boolean; game?: string; algorithm?: string; error?: string }> {
    try {
      const payload: Record<string, unknown> = {}
      if (configPath) payload.config = configPath
      const result = await api('/api/validate-config', payload)
      return result as { ok: boolean; game?: string; algorithm?: string; error?: string }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
  }

  async function runMlSmoke(params: { steps?: number; lr?: number; seed?: number }): Promise<{ ok: boolean; passed?: boolean; improvement?: number; initial_loss?: number; final_loss?: number; error?: string }> {
    try {
      const result = await api('/api/ml-smoke', params as Record<string, unknown>)
      return result as { ok: boolean; passed?: boolean; improvement?: number; initial_loss?: number; final_loss?: number; error?: string }
    } catch (e) {
      return { ok: false, error: e instanceof Error ? e.message : String(e) }
    }
  }

  async function startCollect(params: { experiment?: string; episodes?: number; max_steps?: number; headed?: boolean }) {
    await api('/api/collect-random/start', params as Record<string, unknown>)
    await refresh()
  }

  async function stopCollect() {
    await api('/api/collect-random/stop')
    await refresh()
  }

  async function startWorldTraining(params: { experiment?: string; steps?: number; collect_steps?: number; batch_size?: number; lr?: number; dashboard_every?: number; headed?: boolean }) {
    await api('/api/train-world/start', params as Record<string, unknown>)
    await refresh()
  }

  async function stopWorldTraining() {
    await api('/api/train-world/stop')
    await refresh()
  }

  const evalResult = computed(() => evalJob.value?.result ?? null)

  return {
    summary, latestStep, steps, episodes,
    job, evalJob, worldJob, collectJob, evalResult,
    gameSettings, configDetail, runDir, resetKey, actionKeys,
    isTraining, isEvaluating, isWorldTraining, isCollecting, headerStatus,
    chartPoints, refresh, validateConfig, runMlSmoke,
    startCollect, stopCollect, startWorldTraining, stopWorldTraining,
  }
})

export interface ConfigGroup {
  title: string
  collapsed?: boolean
  fields: [string, unknown, string, { type: string; options?: string[]; min?: number; max?: number; step?: number }][]
}
