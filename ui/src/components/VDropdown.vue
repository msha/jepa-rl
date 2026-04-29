<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: string
  options: { value: string; label: string }[]
  placeholder?: string
  title?: string
  disabled?: boolean
  monospace?: boolean
  compact?: boolean
  fullWidth?: boolean
}>(), {
  placeholder: '',
  title: '',
  disabled: false,
  monospace: true,
  compact: false,
  fullWidth: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
  change: []
}>()

const open = ref(false)
const highlightIdx = ref(-1)
const trigger = ref<HTMLElement | null>(null)
const listbox = ref<HTMLElement | null>(null)

const selectedLabel = computed(() => {
  const opt = props.options.find(o => o.value === props.modelValue)
  return opt ? opt.label : props.placeholder
})

function toggle(e: Event) {
  if (props.disabled) return
  e.stopPropagation()
  open.value = !open.value
  if (open.value) {
    highlightIdx.value = props.options.findIndex(o => o.value === props.modelValue)
    nextTick(() => {
      scrollToHighlighted()
      listbox.value?.focus()
    })
  }
}

function close() {
  open.value = false
  highlightIdx.value = -1
}

function select(val: string) {
  emit('update:modelValue', val)
  emit('change')
  close()
}

function onKeydown(e: KeyboardEvent) {
  if (!open.value) return
  const len = props.options.length
  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault()
      highlightIdx.value = highlightIdx.value < len - 1 ? highlightIdx.value + 1 : 0
      scrollToHighlighted()
      break
    case 'ArrowUp':
      e.preventDefault()
      highlightIdx.value = highlightIdx.value > 0 ? highlightIdx.value - 1 : len - 1
      scrollToHighlighted()
      break
    case 'Enter':
    case ' ':
      e.preventDefault()
      if (highlightIdx.value >= 0 && highlightIdx.value < len) {
        select(props.options[highlightIdx.value].value)
      }
      break
    case 'Escape':
      e.preventDefault()
      close()
      trigger.value?.focus()
      break
  }
}

function scrollToHighlighted() {
  nextTick(() => {
    const el = listbox.value?.querySelector('.vdd-opt--hl') as HTMLElement | null
    el?.scrollIntoView({ block: 'nearest' })
  })
}

function onClickOutside(e: MouseEvent) {
  const root = trigger.value?.closest('.vdd')
  if (root && !root.contains(e.target as Node)) {
    close()
  }
}

onMounted(() => document.addEventListener('click', onClickOutside, true))
onBeforeUnmount(() => document.removeEventListener('click', onClickOutside, true))
</script>

<template>
  <div
    class="vdd"
    :class="{ 'vdd--open': open, 'vdd--disabled': disabled, 'vdd--full': fullWidth }"
    :title="title"
  >
    <button
      ref="trigger"
      class="vdd-trigger"
      :class="{ 'vdd-trigger--mono': monospace, 'vdd-trigger--compact': compact }"
      @click="toggle"
      @keydown="onKeydown"
      aria-haspopup="listbox"
      :aria-expanded="open"
      :disabled="disabled"
      type="button"
    >
      <span class="vdd-text" :class="{ 'vdd-text--placeholder': !options.find(o => o.value === modelValue) }">
        {{ selectedLabel }}
      </span>
      <span class="vdd-chevron" aria-hidden="true">
        <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
          <path d="M1 1L5 5L9 1" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </span>
    </button>
    <Transition name="vdd-list">
      <div v-if="open" class="vdd-list-wrap" @keydown="onKeydown" ref="listbox" tabindex="-1">
        <ul class="vdd-list" role="listbox">
          <li
            v-for="(opt, i) in options"
            :key="opt.value"
            class="vdd-opt"
            :class="{
              'vdd-opt--sel': opt.value === modelValue,
              'vdd-opt--hl': i === highlightIdx,
            }"
            @click.stop="select(opt.value)"
            @mouseenter="highlightIdx = i"
            role="option"
            :aria-selected="opt.value === modelValue"
          >
            <span class="vdd-opt-text">{{ opt.label }}</span>
            <span v-if="opt.value === modelValue" class="vdd-opt-check" aria-hidden="true">✓</span>
          </li>
        </ul>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.vdd {
  position: relative;
  display: inline-flex;
  flex: 0 1 auto;
  min-width: 0;
}

.vdd--full {
  flex: 1 1 0;
  width: 100%;
}

/* Trigger button */
.vdd-trigger {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--text);
  padding: 3px 8px 3px 6px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  cursor: pointer;
  outline: none;
  transition: border-color 0.15s, box-shadow 0.15s;
  line-height: 1.4;
}

.vdd-trigger--compact {
  padding: 2px 5px 2px 4px;
  font-size: 10px;
  gap: 4px;
}

.vdd-trigger--compact .vdd-chevron svg {
  width: 8px;
  height: 5px;
}

.vdd-trigger--compact .vdd-text {
  max-width: 110px;
}

.vdd-trigger:not(:disabled):hover {
  border-color: var(--muted);
}

.vdd-trigger:not(:disabled):focus,
.vdd--open > .vdd-trigger {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(196, 145, 82, 0.2);
}

.vdd--disabled > .vdd-trigger {
  opacity: 0.4;
  cursor: not-allowed;
}

/* Text area */
.vdd-text {
  flex: 1 1 0;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: left;
}

.vdd-text--placeholder {
  color: var(--muted);
  font-style: italic;
  opacity: 0.6;
}

/* Chevron */
.vdd-chevron {
  flex-shrink: 0;
  color: var(--muted);
  transition: transform 0.2s ease, color 0.15s;
  display: flex;
  align-items: center;
}

.vdd--open > .vdd-trigger .vdd-chevron {
  transform: rotate(180deg);
  color: var(--accent);
}

/* Dropdown list */
.vdd-list-wrap {
  position: absolute;
  top: calc(100% + 3px);
  left: 0;
  right: 0;
  min-width: 100%;
  z-index: 300;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.55), 0 0 0 1px rgba(196, 145, 82, 0.06);
  overflow: hidden;
}

.vdd-list {
  list-style: none;
  max-height: 220px;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 3px 0;
  margin: 0;
}

.vdd-list::-webkit-scrollbar {
  width: 3px;
}

.vdd-list::-webkit-scrollbar-track {
  background: transparent;
}

.vdd-list::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}

/* Option */
.vdd-opt {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px 4px 8px;
  cursor: pointer;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: var(--text);
  transition: background 0.1s;
  line-height: 1.5;
}

.vdd-opt--hl {
  background: var(--surface-2);
}

.vdd-opt--sel {
  color: var(--accent);
}

.vdd-opt-text {
  flex: 1 1 0;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.vdd-opt-check {
  flex-shrink: 0;
  font-size: 9px;
  color: var(--accent);
  opacity: 0.8;
}

.vdd-opt--hl.vdd-opt--sel {
  background: rgba(196, 145, 82, 0.1);
}

/* Transition */
.vdd-list-enter-active {
  transition: opacity 0.12s ease, transform 0.12s ease;
}

.vdd-list-leave-active {
  transition: opacity 0.08s ease, transform 0.08s ease;
}

.vdd-list-enter-from,
.vdd-list-leave-to {
  opacity: 0;
  transform: translateY(-3px);
}
</style>
