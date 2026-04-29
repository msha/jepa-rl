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
}

export const useRunsStore = defineStore('runs', () => {
  const runs = ref<Run[]>([])
  const selectedRun = ref('')
  const checkpoints = ref<Checkpoint[]>([])
  const runDir = ref('')
  const runConfigDetail = ref<ConfigGroup[] | null>(null)
  const showSmoke = ref(false)

  async function loadRuns() {
    try {
      const data = await getJson<{ runs: Run[] }>('/api/runs' + (showSmoke.value ? '?smoke=true' : ''))
      runs.value = data.runs || []
      if (!selectedRun.value && runs.value.length > 0) {
        const first = runs.value[0].name
        selectedRun.value = first
        loadRunDetail(first)
      }
    } catch { /* swallow */ }
  }

  async function loadRunDetail(name: string) {
    if (!name) {
      checkpoints.value = []
      runDir.value = ''
      runConfigDetail.value = null
      return
    }
    selectedRun.value = name
    try {
      const data = await getJson<RunDetail>(`/api/run-detail?name=${encodeURIComponent(name)}`)
      checkpoints.value = data.checkpoints || []
      runDir.value = data.run_dir || ''
      runConfigDetail.value = data.detail || null
    } catch { /* swallow */ }
  }

  async function deleteRun(name: string) {
    await api('/api/delete-run', { name })
    if (selectedRun.value === name) {
      selectedRun.value = ''
      checkpoints.value = []
      runDir.value = ''
      runConfigDetail.value = null
    }
    await loadRuns()
  }

  return { runs, selectedRun, checkpoints, runDir, runConfigDetail, showSmoke, loadRuns, loadRunDetail, deleteRun }
})
