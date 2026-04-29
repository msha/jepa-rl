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
const gameConfigGroups = new Set(['game'])
const configGroups = computed(() => {
  const baseGroups = training.baseConfigDetail.length ? training.baseConfigDetail : training.configDetail
  const groups = runs.runConfigDetail ?? baseGroups
  if (runs.runConfigDetail) return groups.map(readonlyGroup)
  return groups.filter(g => gameConfigGroups.has(g.title))
})
const panelTitle = computed(() => runs.runConfigDetail ? 'run snapshot' : 'game config')

function readonlyGroup(group: any) {
  return {
    ...group,
    fields: group.fields.map((field: any[]) => [
      field[0],
      field[1],
      field[2],
      { ...(field[3] || {}), type: 'readonly' },
    ]),
  }
}

function toggleGroup(title: string) {
  if (collapsedGroups.value.has(title)) collapsedGroups.value.delete(title)
  else collapsedGroups.value.add(title)
}

function isCollapsed(title: string): boolean {
  return collapsedGroups.value.has(title)
}

async function applyOverride(group: string, key: string, value: string) {
  if (training.isTraining) throw new Error('Cannot change config while training')
  await config.applyOverrides([{ group, key, value }])
  await training.refresh()
}

// Popover state
interface EditField {
  group: string
  groupLabel: string
  key: string
  fieldLabel: string
  type: string
  opts?: string[]
  min?: number
  max?: number
  step?: number
}

const editingField = ref<EditField | null>(null)
const editValue = ref<string>('')
const editPopoverStyle = ref<Record<string, string>>({})
const saveError = ref('')
const saving = ref(false)

const selectFieldOptions = computed(() => {
  if (!editingField.value?.opts) return []
  return editingField.value.opts.map(o => ({ value: String(o), label: String(o) }))
})

const editValueAsNum = computed({
  get: () => {
    const n = parseFloat(editValue.value)
    return isNaN(n) ? 0 : n
  },
  set: (v: number) => { editValue.value = String(v) }
})

function openEdit(group: string, groupLabel: string, f: any[], event: MouseEvent) {
  if (runs.runConfigDetail || f[3]?.type === 'readonly') return
  if (training.isTraining) return
  const meta = f[3] || {}
  editingField.value = {
    group,
    groupLabel: groupLabel || group,
    key: f[0],
    fieldLabel: meta.label || f[0],
    type: meta.type || 'text',
    opts: meta.options,
    min: meta.min,
    max: meta.max,
    step: meta.step,
  }
  // For grayscale select, show semantic value not raw bool
  if (group === 'observation' && f[0] === 'grayscale') {
    editValue.value = String(f[1])
  } else {
    editValue.value = String(f[1])
  }
  saveError.value = ''
  saving.value = false
  const el = event.currentTarget as HTMLElement
  const rect = el.getBoundingClientRect()
  const w = Math.max(240, rect.width)
  let top = rect.bottom + 4
  let left = rect.left
  if (top + 160 > window.innerHeight - 8) top = rect.top - 160 - 4
  if (left + w > window.innerWidth - 8) left = rect.right - w
  if (left < 8) left = 8
  editPopoverStyle.value = { top: `${top}px`, left: `${left}px`, width: `${w}px` }
}

function closeEdit() {
  editingField.value = null
  saveError.value = ''
}

function validate(): string | null {
  if (!editingField.value) return 'No field selected'
  const f = editingField.value
  if (f.type === 'number') {
    const n = parseFloat(editValue.value)
    if (isNaN(n)) return 'Value must be a number'
    if (f.min != null && n < f.min) return `Minimum value is ${f.min}`
    if (f.max != null && n > f.max) return `Maximum value is ${f.max}`
  }
  if (f.type === 'text' && !editValue.value.trim()) return 'Value cannot be empty'
  return null
}

function decrement() {
  const n = parseFloat(editValue.value)
  if (isNaN(n)) return
  const s = editingField.value?.step ?? 1
  const newVal = Math.round((n - s) * 10000) / 10000
  const min = editingField.value?.min
  editValue.value = String(min != null ? Math.max(min, newVal) : newVal)
}

function increment() {
  const n = parseFloat(editValue.value)
  if (isNaN(n)) return
  const s = editingField.value?.step ?? 1
  const newVal = Math.round((n + s) * 10000) / 10000
  const max = editingField.value?.max
  editValue.value = String(max != null ? Math.min(max, newVal) : newVal)
}

async function saveEdit() {
  if (!editingField.value) return
  const err = validate()
  if (err) { saveError.value = err; return }
  saveError.value = ''
  saving.value = true
  try {
    await applyOverride(editingField.value.group, editingField.value.key, editValue.value)
    closeEdit()
  } catch (e) {
    saveError.value = e instanceof Error ? e.message : String(e)
  } finally {
    saving.value = false
  }
}

function onCheckboxChange(e: Event, group: string, key: string) {
  const el = e.target as HTMLInputElement
  if (!el) return
  applyOverride(group, key, el.checked ? 'true' : 'false').catch(console.error)
}
</script>

<template>
  <div class="config-panel">
    <div class="section-header" title="All config parameters for the active game" style="margin-bottom: 6px;">
      {{ panelTitle }}
    </div>
    <div class="run-config" id="runConfig" style="display: flex; flex-direction: column; gap: 8px;">
      <div v-for="group in configGroups" :key="group.title" class="cfg-group">
        <div class="cfg-group-title" @click="toggleGroup(group.title)" style="font-size: 9px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); cursor: pointer; margin-bottom: 3px; display: flex; align-items: center; gap: 4px;">
          <span style="font-size: 8px; opacity: 0.7;">{{ isCollapsed(group.title) || group.collapsed ? '▶' : '▼' }}</span> {{ group.title }}
        </div>
        <div class="settings-table" v-show="!isCollapsed(group.title) && !group.collapsed">
          <template v-for="f in group.fields" :key="f[0]">
            <span class="sk">{{ f[0] }}</span>
            <div v-if="f[3]?.type === 'readonly'" class="sv">
              <span class="ro-val">{{ String(f[1]) }}</span>
              <span v-if="f[2]" class="cfg-desc" :data-tip="f[2]" :data-short="f[2]"></span>
            </div>
            <div v-else-if="f[3]?.type === 'bool'" class="sv">
              <input type="checkbox" :checked="!!f[1]" @change="e => onCheckboxChange(e, group.title, f[0])" style="accent-color: var(--accent); margin: 0; flex-shrink: 0;">
              <span v-if="f[2]" class="cfg-desc" :data-tip="f[2]" :data-short="f[2]"></span>
            </div>
            <div v-else class="sv">
              <span class="clickable-field" @click="openEdit(group.title, (group as any).group_label, f, $event)">{{ String(f[1]) }}</span>
              <span v-if="f[2]" class="cfg-desc" :data-tip="f[2]" :data-short="f[2]"></span>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- Inline Edit Popover -->
    <div v-if="editingField" class="popover-backdrop" @click="closeEdit"></div>
    <div v-if="editingField" class="popover" :style="editPopoverStyle">
      <div class="popover-breadcrumb">
        <span class="breadcrumb-group">{{ editingField.groupLabel }}</span>
        <span class="breadcrumb-sep">&#8250;</span>
        <span class="breadcrumb-field">{{ editingField.fieldLabel }}</span>
      </div>

      <!-- Select dropdown -->
      <VDropdown
        v-if="editingField.type === 'select'"
        v-model="editValue"
        :options="selectFieldOptions"
        @change="saveEdit"
        full-width
      />

      <!-- Number input with +/- buttons -->
      <div v-else-if="editingField.type === 'number'" class="num-input-row">
        <button class="num-btn" @click="decrement" title="Decrement">&minus;</button>
        <input v-model="editValue"
               inputmode="decimal"
               class="popover-input num-input"
               @keyup.enter="saveEdit"
               @keyup.escape="closeEdit"
               autofocus>
        <button class="num-btn" @click="increment" title="Increment">&plus;</button>
      </div>
      <!-- Slider for ranged numeric values -->
      <div v-if="editingField.type === 'number' && editingField.min != null && editingField.max != null" class="slider-row">
        <input type="range"
               :min="editingField.min"
               :max="editingField.max"
               :step="editingField.step || 0.01"
               v-model.number="editValueAsNum"
               class="cfg-slider">
        <span class="slider-bounds">{{ editingField.min }}</span>
        <span class="slider-bounds">{{ editingField.max }}</span>
      </div>

      <!-- Text input -->
      <input v-if="editingField.type !== 'select' && editingField.type !== 'number'"
             v-model="editValue"
             type="text"
             class="popover-input"
             @keyup.enter="saveEdit"
             @keyup.escape="closeEdit"
             autofocus>

      <!-- Error display -->
      <div v-if="saveError" class="popover-error">{{ saveError }}</div>

      <div class="popover-actions">
        <button class="btn-tiny" @click="closeEdit">cancel</button>
        <button class="btn-accent-tiny" @click="saveEdit" :disabled="saving">{{ saving ? 'saving...' : 'save' }}</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.cfg-group + .cfg-group {
  margin-top: 4px;
}

.settings-table .sv {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.settings-table .sv .ro-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--text);
  font-weight: 500;
  flex-shrink: 0;
}

.settings-table .sv .clickable-field {
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  line-height: 1;
  color: var(--accent);
  font-weight: 500;
  transition: all 0.15s;
  border-bottom: 1px solid rgba(196, 145, 82, 0.3);
  flex-shrink: 0;
}
.settings-table .sv .clickable-field:hover {
  border-bottom-color: var(--accent);
}

.cfg-desc {
  position: relative;
  cursor: help;
  flex: 1;
  min-width: 0;
  line-height: 1.4;
  display: flex;
  font-size: 0;
}
.cfg-desc::before {
  content: attr(data-short);
  display: block;
  font-size: 9px;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  min-width: 0;
}
.cfg-desc::after {
  content: attr(data-tip);
  position: absolute;
  left: 0;
  bottom: calc(100% + 6px);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 6px 10px;
  font-size: 10px;
  line-height: 1.5;
  color: var(--text);
  white-space: pre-line;
  width: max-content;
  max-width: 300px;
  pointer-events: none;
  opacity: 0;
  transform: translateY(4px);
  transition: opacity 0.15s, transform 0.15s;
  z-index: 100;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.cfg-desc:hover::after {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

/* Numeric input row */
.num-input-row {
  display: flex;
  gap: 4px;
  align-items: center;
}
.num-btn {
  width: 26px;
  height: 26px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--accent);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.15s;
  flex-shrink: 0;
  line-height: 1;
  padding: 0;
  font-family: 'IBM Plex Mono', monospace;
}
.num-btn:hover {
  background: var(--surface-2);
  border-color: var(--accent);
}
.num-btn:active {
  background: rgba(196, 145, 82, 0.12);
  transform: scale(0.95);
}
.num-input {
  flex: 1;
  text-align: center;
}

/* Slider */
.slider-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 6px;
}
.cfg-slider {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  outline: none;
}
.cfg-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: none;
  box-shadow: 0 0 4px rgba(196, 145, 82, 0.4);
}
.cfg-slider::-moz-range-thumb {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--accent);
  cursor: pointer;
  border: none;
}
.slider-bounds {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: var(--muted);
  flex-shrink: 0;
}

/* Error display */
.popover-error {
  color: var(--red);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  padding: 4px 0 0;
  line-height: 1.4;
}
</style>
