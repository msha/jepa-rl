<script setup lang="ts">
import { ref } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useConfigStore } from '../stores/config'

const training = useTrainingStore()
const config = useConfigStore()

const collapsedGroups = ref<Set<string>>(new Set())

function toggleGroup(title: string) {
  if (collapsedGroups.value.has(title)) collapsedGroups.value.delete(title)
  else collapsedGroups.value.add(title)
}

function isCollapsed(title: string): boolean {
  return collapsedGroups.value.has(title)
}

async function applyEdits() {
  const overrides: { group: string; key: string; value: string }[] = []
  const container = document.getElementById('runConfig')
  if (!container) return

  container.querySelectorAll<HTMLInputElement | HTMLSelectElement>('.cfg-input, .cfg-select').forEach(el => {
    overrides.push({
      group: (el as HTMLElement).dataset.group || '',
      key: (el as HTMLElement).dataset.key || '',
      value: el.value,
    })
  })
  container.querySelectorAll<HTMLInputElement>('.cfg-checkbox').forEach(el => {
    overrides.push({
      group: el.dataset.group || '',
      key: el.dataset.key || '',
      value: el.checked ? 'true' : 'false',
    })
  })

  if (!overrides.length) return
  try {
    await config.applyOverrides(overrides)
    training.refresh()
  } catch (e) {
    console.error(e)
  }
}
</script>

<template>
  <div class="config-panel">
    <div class="section-header" title="All config parameters for the active game">
      config &nbsp;<button @click="applyEdits" class="cfg-apply-btn" title="Apply edited config values now">apply</button>
    </div>
    <div class="run-config" id="runConfig">
      <div v-for="group in training.configDetail" :key="group.title"
           :class="['cfg-group', { 'cfg-collapsed': isCollapsed(group.title) || group.collapsed }]"
           :data-group-title="group.title">
        <div class="cfg-group-title" @click="toggleGroup(group.title)">{{ group.title }}</div>
        <div class="cfg-fields">
          <template v-for="f in group.fields" :key="f[0]">
            <div v-if="f[3]?.type === 'readonly'" class="cfg-row">
              <span class="cfg-key" :title="f[2]">{{ f[0] }}</span>
              <span class="cfg-val">{{ String(f[1]) }}</span>
            </div>
            <div v-else-if="f[3]?.type === 'bool'" class="cfg-row">
              <span class="cfg-key" :title="f[2]">{{ f[0] }}</span>
              <input type="checkbox" class="cfg-checkbox" :data-group="group.title" :data-key="f[0]" :checked="!!f[1]" :title="f[2]">
            </div>
            <div v-else-if="f[3]?.type === 'select'" class="cfg-row">
              <span class="cfg-key" :title="f[2]">{{ f[0] }}</span>
              <select class="cfg-select" :data-group="group.title" :data-key="f[0]" data-type="select" :title="f[2]">
                <option v-for="o in f[3].options || []" :key="o" :value="o" :selected="String(o) === String(f[1])">{{ o }}</option>
              </select>
            </div>
            <div v-else class="cfg-row">
              <span class="cfg-key" :title="f[2]">{{ f[0] }}</span>
              <input class="cfg-input"
                     :type="f[3]?.type === 'number' ? 'number' : 'text'"
                     :min="f[3]?.min"
                     :max="f[3]?.max"
                     :step="f[3]?.step ?? 1"
                     :data-group="group.title"
                     :data-key="f[0]"
                     :data-type="f[3]?.type || 'text'"
                     :value="String(f[1])"
                     :title="f[2]">
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
