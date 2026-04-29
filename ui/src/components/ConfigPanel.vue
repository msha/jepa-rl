<script setup lang="ts">
import { ref, computed } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useConfigStore } from '../stores/config'
import { useRunsStore } from '../stores/runs'
import VDropdown from './VDropdown.vue'

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

// Modal State
const editingField = ref<{ group: string, key: string, val: unknown, type: string, opts?: string[] } | null>(null)
const editValue = ref<string>('')
const editPopoverStyle = ref<Record<string, string>>({})

const selectFieldOptions = computed(() => {
  if (!editingField.value?.opts) return []
  return editingField.value.opts.map(o => ({ value: String(o), label: String(o) }))
})

function openEdit(group: string, f: any[], event: MouseEvent) {
  if (f[3]?.type === 'readonly') return
  editingField.value = {
    group,
    key: f[0],
    val: f[1],
    type: f[3]?.type || 'text',
    opts: f[3]?.options
  }
  editValue.value = String(f[1])
  const el = event.currentTarget as HTMLElement
  const rect = el.getBoundingClientRect()
  const w = Math.max(220, rect.width)
  let top = rect.bottom + 4
  let left = rect.left
  if (top + 110 > window.innerHeight - 8) top = rect.top - 110 - 4
  if (left + w > window.innerWidth - 8) left = rect.right - w
  if (left < 8) left = 8
  editPopoverStyle.value = { top: `${top}px`, left: `${left}px`, width: `${w}px` }
}

function closeEdit() {
  editingField.value = null
}

function saveEdit() {
  if (!editingField.value) return
  applyOverride(editingField.value.group, editingField.value.key, editValue.value)
  closeEdit()
}

function onCheckboxChange(e: Event, group: string, key: string) {
  const el = e.target as HTMLInputElement
  if (!el) return
  applyOverride(group, key, el.checked ? 'true' : 'false')
}
</script>

<template>
  <div class="config-panel">
    <div class="section-header" title="All config parameters for the active game" style="margin-bottom: 6px;">
      config
    </div>
    <div class="run-config" id="runConfig" style="display: flex; flex-direction: column; gap: 8px;">
      <div v-for="group in configGroups" :key="group.title" class="cfg-group">
        <div class="cfg-group-title" @click="toggleGroup(group.title)" style="font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); cursor: pointer; margin-bottom: 3px; display: flex; align-items: center; gap: 4px;">
          <span style="font-size: 8px; opacity: 0.7;">{{ isCollapsed(group.title) || group.collapsed ? '▶' : '▼' }}</span> {{ group.title }}
        </div>
        <div class="settings-table" v-show="!isCollapsed(group.title) && !group.collapsed">
          <template v-for="f in group.fields" :key="f[0]">
            <span class="sk" :title="f[2]" style="display: flex; align-items: center;">{{ f[0] }}</span>
            <span v-if="f[3]?.type === 'readonly'" class="sv" :title="f[2]">{{ String(f[1]) }}</span>
            <div v-else-if="f[3]?.type === 'bool'" style="display: flex; align-items: center;">
              <input type="checkbox" :checked="!!f[1]" :title="f[2]" @change="e => onCheckboxChange(e, group.title, f[0])" style="accent-color: var(--accent); margin: 0;">
            </div>
            <div v-else class="sv clickable-field" :title="f[2] + ' (Click to edit)'" @click="openEdit(group.title, f, $event)">
              {{ String(f[1]) }}
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- Inline Edit Popover -->
    <div v-if="editingField" class="popover-backdrop" @click="closeEdit"></div>
    <div v-if="editingField" class="popover" :style="editPopoverStyle">
      <div class="popover-label">{{ editingField.key }}</div>

      <VDropdown
        v-if="editingField.type === 'select'"
        v-model="editValue"
        :options="selectFieldOptions"
        @change="saveEdit"
        full-width
      />

      <input v-else
             v-model="editValue"
             :type="editingField.type === 'number' ? 'number' : 'text'"
             class="popover-input"
             @keyup.enter="saveEdit"
             @keyup.escape="closeEdit"
             autofocus>

      <div class="popover-actions">
        <button class="btn-tiny" @click="closeEdit">cancel</button>
        <button class="btn-accent-tiny" @click="saveEdit">save</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cfg-group + .cfg-group {
  margin-top: 4px;
}
</style>
