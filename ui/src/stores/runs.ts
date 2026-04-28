import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getJson } from '../api/client'

interface Run {
  name: string
  steps?: number
  episodes?: number
  best_score?: number
  algorithm?: string
  checkpoints: string[]
}

interface RunDetail {
  detail: unknown[]
  summary: Record<string, unknown>
  checkpoints: string[]
  run_dir: string
}

export const useRunsStore = defineStore('runs', () => {
  const runs = ref<Run[]>([])
  const selectedRun = ref('')
  const checkpoints = ref<string[]>([])
  const runDir = ref('')

  async function loadRuns() {
    try {
      const data = await getJson<{ runs: Run[] }>('/api/runs')
      runs.value = data.runs || []
    } catch { /* swallow */ }
  }

  async function loadRunDetail(name: string) {
    if (!name) { checkpoints.value = []; runDir.value = ''; return }
    selectedRun.value = name
    try {
      const data = await getJson<RunDetail>(`/api/run-detail?name=${encodeURIComponent(name)}`)
      checkpoints.value = data.checkpoints || []
      runDir.value = data.run_dir || ''
    } catch { /* swallow */ }
  }

  return { runs, selectedRun, checkpoints, runDir, loadRuns, loadRunDetail }
})
