<script setup lang="ts">
import { ref, computed } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useConfigStore } from '../stores/config'
import { useRunsStore } from '../stores/runs'

const training = useTrainingStore()
const config = useConfigStore()
const runs = useRunsStore()

const collapsedGroups = ref<Set<string>>(new Set())

const configGroups = computed(() => runs.runConfigDetail ?? training.configDetail)

function toggleGroup(title: string) {
  if (collapsedGroups.value.has(title)) collapsedGroups.value.delete(title)
  else collapsedGroups.value.add(title)
}

function isCollapsed(title: string): boolean {
  return collapsedGroups.value.has(title)
}

async function applyOverride(group: string, key: string, value: string) {
  if (training.isTraining) return
  try {
    await config.applyOverrides([{ group, key, value }])
    training.refresh()
  } catch (e) {
    console.error(e)
  }
}

function onInputBlur(e: FocusEvent) {
  const el = e.target as HTMLInputElement
  if (!el) return
  const group = el.dataset.group || ''
  const key = el.dataset.key || ''
  applyOverride(group, key, el.value)
}

function onInputEnter(e: KeyboardEvent) {
  if (e.key !== 'Enter') return
  const el = e.target as HTMLInputElement
  if (!el) return
  el.blur()
}

function onSelectChange(e: Event) {
  const el = e.target as HTMLSelectElement
  if (!el) return
  applyOverride(el.dataset.group || '', el.dataset.key || '', el.value)
}

function onCheckboxChange(e: Event) {
  const el = e.target as HTMLInputElement
  if (!el) return
  applyOverride(el.dataset.group || '', el.dataset.key || '', el.checked ? 'true' : 'false')
}
</script>

<template>
  <div class="config-panel">
    <div class="section-header" title="All config parameters for the active game">
      config
    </div>
    <div class="run-config" id="runConfig">
      <div v-for="group in configGroups" :key="group.title"
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
              <input type="checkbox" class="cfg-checkbox" :data-group="group.title" :data-key="f[0]" :checked="!!f[1]" :title="f[2]" @change="onCheckboxChange">
            </div>
            <div v-else-if="f[3]?.type === 'select'" class="cfg-row">
              <span class="cfg-key" :title="f[2]">{{ f[0] }}</span>
              <select class="cfg-select" :data-group="group.title" :data-key="f[0]" data-type="select" :title="f[2]" @change="onSelectChange">
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
                     :title="f[2]"
                     @blur="onInputBlur"
                     @keydown="onInputEnter">
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>
