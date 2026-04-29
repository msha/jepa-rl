import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, getJson } from '../api/client'
import type { ConfigGroup } from './training'

interface Checkpoint {
  file: string
  label: string
}

interface Run {
  name: string
  experiment_name?: string
  steps?: number
  episodes?: number
  best_score?: number
  algorithm?: string
  checkpoints: Checkpoint[]
}

interface RunDetail {
  detail: ConfigGroup[]
  summary: Record<string, unknown>
  checkpoints: Checkpoint[]
  run_dir: string
  model_info?: Record<string, unknown>
}

interface CollectedDataset {
  name: string
  episodes: number
  mean_score: number | null
  max_score: number | null
  min_score: number | null
  median_score: number | null
  mean_length: number | null
  total_steps: number
  size_bytes: number
}

export const useRunsStore = defineStore('runs', () => {
  const runs = ref<Run[]>([])
  const selectedRun = ref('')
  const checkpoints = ref<Checkpoint[]>([])
  const runDir = ref('')
  const runConfigDetail = ref<ConfigGroup[] | null>(null)
  const runModelInfo = ref<Record<string, unknown> | null>(null)
  const showSmoke = ref(false)
  const collectedDatasets = ref<CollectedDataset[]>([])

  async function loadRuns() {
    try {
      const data = await getJson<{ runs: Run[] }>('/api/runs' + (showSmoke.value ? '?smoke=true' : ''))
      runs.value = data.runs || []
    } catch { /* swallow */ }
  }

  async function loadRunDetail(name: string) {
    if (!name) {
      clearSelection()
      return
    }
    selectedRun.value = name
    try {
      const selected = await api('/api/run/select', { name })
      const data = selected as unknown as RunDetail
      checkpoints.value = data.checkpoints || []
      runDir.value = data.run_dir || ''
      runConfigDetail.value = data.detail || null
      runModelInfo.value = data.model_info || null
    } catch {
      try {
        const data = await getJson<RunDetail>(`/api/run-detail?name=${encodeURIComponent(name)}`)
        checkpoints.value = data.checkpoints || []
        runDir.value = data.run_dir || ''
        runConfigDetail.value = data.detail || null
        runModelInfo.value = data.model_info || null
      } catch { /* swallow */ }
    }
  }

  async function createRun(name: string, overrides: { group: string; key: string; value: string }[]) {
    const data = await api('/api/run/create', { name, overrides })
    selectedRun.value = name
    checkpoints.value = (data.checkpoints as Checkpoint[]) || []
    runDir.value = String(data.run_dir || '')
    runConfigDetail.value = (data.detail as ConfigGroup[]) || null
    runModelInfo.value = (data.model_info as Record<string, unknown>) || null
    await loadRuns()
  }

  function clearSelection() {
    selectedRun.value = ''
    checkpoints.value = []
    runDir.value = ''
    runConfigDetail.value = null
    runModelInfo.value = null
  }

  async function loadCollectedDatasets() {
    try {
      const data = await getJson<{ datasets: CollectedDataset[] }>('/api/collected-datasets')
      collectedDatasets.value = data.datasets || []
    } catch { /* swallow */ }
  }

  async function deleteRun(name: string) {
    await api('/api/delete-run', { name })
    if (selectedRun.value === name) {
      clearSelection()
    }
    await loadRuns()
    await loadCollectedDatasets()
  }

  async function deleteDataset(name: string) {
    await api('/api/delete-run', { name })
    await loadCollectedDatasets()
  }

  return {
    runs,
    selectedRun,
    checkpoints,
    runDir,
    runConfigDetail,
    runModelInfo,
    showSmoke,
    collectedDatasets,
    loadRuns,
    loadRunDetail,
    createRun,
    clearSelection,
    loadCollectedDatasets,
    deleteRun,
    deleteDataset,
  }
})
