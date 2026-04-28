import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api, getJson } from '../api/client'

export const useConfigStore = defineStore('config', () => {
  const configs = ref<{ path: string; name: string }[]>([])
  const currentPath = ref('')
  const configGame = ref('')

  async function loadConfigs() {
    try {
      const data = await getJson<{ configs: { path: string; name: string }[] }>('/api/configs')
      configs.value = data.configs || []
      if (!currentPath.value) {
        try {
          const st = await getJson<{ config: { path: string } }>('/api/state')
          if (st.config?.path) currentPath.value = st.config.path
        } catch { /* */ }
      }
    } catch { /* swallow */ }
  }

  async function switchConfig(path: string) {
    const res = await api('/api/switch-config', { config: path })
    currentPath.value = path
    configGame.value = (res.game as string) || ''
  }

  async function applyOverrides(overrides: { group: string; key: string; value: string }[]) {
    await api('/api/update-config', { overrides })
  }

  return { configs, currentPath, configGame, loadConfigs, switchConfig, applyOverrides }
})
