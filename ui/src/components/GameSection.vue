<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { usePolling } from '../composables/usePolling'
import type { Job, EvalJob } from '../stores/training'

const props = defineProps<{
  job: Job | null
  evalJob: EvalJob | null
  resetKey: string
  actionKeys: string[]
  steps: Record<string, unknown>[]
}>()

const isTraining = computed(() => !!props.job?.running || props.job?.status === 'running' || props.job?.status === 'starting')
const isEvaluating = computed(() => !isTraining.value && (!!props.evalJob?.running || props.evalJob?.status === 'running'))

const showGame = ref(true)
const gameStats = ref('')
const gameFrame = ref<HTMLIFrameElement | null>(null)
const gameReloadToken = ref(0)
const evalClientError = ref('')

let aiInterval: ReturnType<typeof window.setInterval> | null = null
let aiStepInFlight = false
let lastMirroredTrainingStep = 0
let trainingPlaybackTimers: ReturnType<typeof window.setTimeout>[] = []

const gameTitle = computed(() => {
  if (isTraining.value) return 'training'
  if (isEvaluating.value) return 'AI playing'
  if (props.evalJob?.result) return 'eval done'
  return 'game view'
})

const gameStatus = computed(() => {
  if (evalClientError.value) return `eval error: ${evalClientError.value}`
  if (isTraining.value) return `training · step ${props.job?.requested_steps ?? 0}`
  if (isEvaluating.value) {
    const episode = (props.evalJob?.episode_count ?? 0) + 1
    const target = props.evalJob?.episodes_target ?? 0
    const suffix = target ? ` · ep ${episode}/${target}` : ''
    return `AI playing live${suffix}`
  }
  if (props.job?.status === 'completed' || props.job?.status === 'stopped') return `${props.job.status}`
  if (props.job?.status === 'error') return `error: ${props.job.error || ''}`
  if (props.evalJob?.status === 'error') return `eval error: ${props.evalJob.error || ''}`
  if (props.evalJob?.result) {
    const r = props.evalJob.result
    return `eval · mean ${fmtNum(r.mean_score)} · best ${fmtNum(r.best_score)}`
  }
  return 'manual play'
})

const sectionClass = computed(() => ({
  'game-section': true,
  training: isTraining.value,
  evaluating: isEvaluating.value,
}))

const trainFrameSrc = ref('')
const gameSrc = computed(() => {
  const suffix = gameReloadToken.value ? `&ts=${gameReloadToken.value}` : ''
  return `/game?embed${suffix}`
})

usePolling(async () => {
  // Poll frame
  if (isTraining.value) {
    trainFrameSrc.value = `/api/frame?ts=${Date.now()}`
  }
}, 500)

usePolling(async () => {
  // Poll game stats from iframe
  try {
    const doc = gameFrame.value?.contentDocument
    if (!doc) return
    const score = doc.getElementById('score')
    const lives = doc.getElementById('lives')
    if (score && lives) {
      gameStats.value = `score <strong>${score.textContent || '0'}</strong> &nbsp; lives <strong>${lives.textContent || '3'}</strong>`
    }
  } catch { /* cross-origin */ }
}, 500)

watch(isEvaluating, active => {
  if (active) {
    startAiLoop()
  } else {
    stopAiLoop(false)
  }
}, { immediate: true })

watch(isTraining, active => {
  clearTrainingPlayback()
  if (active) {
    lastMirroredTrainingStep = 0
    reloadGame()
    scheduleEpisodeStart()
  }
}, { immediate: true })

watch(() => props.steps, mirrorTrainingSteps, { deep: true })

onBeforeUnmount(() => {
  stopAiLoop(true)
  clearTrainingPlayback()
})

function toggleGame() {
  showGame.value = !showGame.value
}

function gameAction(action: string) {
  const win = gameFrame.value?.contentWindow
  const doc = gameFrame.value?.contentDocument
  if (!win || !doc) return
  const keyMap: Record<string, { key: string; code: string }> = {
    left: { key: 'ArrowLeft', code: 'ArrowLeft' },
    right: { key: 'ArrowRight', code: 'ArrowRight' },
    space: { key: ' ', code: 'Space' },
    reset: { key: 'r', code: 'KeyR' },
  }
  const mapping = keyMap[action]
  if (!mapping) return
  dispatchKey(win, doc, 'keydown', mapping.key, mapping.code)
  if (action !== 'reset') {
    setTimeout(() => {
      dispatchKey(win, doc, 'keyup', mapping.key, mapping.code)
    }, 120)
  }
}

function startAiLoop() {
  evalClientError.value = ''
  stopAiLoop(false)
  reloadGame()
  scheduleEpisodeStart()
  window.setTimeout(() => {
    if (!isEvaluating.value || aiInterval !== null) return
    aiInterval = window.setInterval(runAiStep, 150)
  }, 750)
}

function stopAiLoop(notifyServer: boolean) {
  if (aiInterval !== null) {
    window.clearInterval(aiInterval)
    aiInterval = null
  }
  aiStepInFlight = false
  if (notifyServer) {
    fetch('/api/eval/stop', { method: 'POST' }).catch(() => {})
  }
}

function mirrorTrainingSteps() {
  if (!isTraining.value || !props.actionKeys.length) return
  const unseen = props.steps
    .filter(step => typeof step.step === 'number' && step.step > lastMirroredTrainingStep)
    .sort((a, b) => Number(a.step) - Number(b.step))
  if (!unseen.length) return

  const toReplay = unseen.length > 24 ? unseen.slice(-24) : unseen
  lastMirroredTrainingStep = Number(unseen[unseen.length - 1].step)
  toReplay.forEach((step, index) => {
    const timer = window.setTimeout(() => mirrorTrainingStep(step), index * 70)
    trainingPlaybackTimers.push(timer)
  })
}

function mirrorTrainingStep(step: Record<string, unknown>) {
  if (!isTraining.value) return
  if (step.done) {
    reloadGame()
    scheduleEpisodeStart()
    return
  }
  const actionIndex = typeof step.action === 'number' ? step.action : Number(step.action)
  if (!Number.isFinite(actionIndex)) return
  const action = props.actionKeys[actionIndex]
  if (!action || action === 'noop') return
  const win = gameFrame.value?.contentWindow
  const doc = gameFrame.value?.contentDocument
  if (!win || !doc) return
  applyKeyAction(win, doc, action)
}

function clearTrainingPlayback() {
  trainingPlaybackTimers.forEach(timer => window.clearTimeout(timer))
  trainingPlaybackTimers = []
}

async function runAiStep() {
  if (aiStepInFlight || !isEvaluating.value) return
  aiStepInFlight = true
  try {
    const iframe = gameFrame.value
    const doc = iframe?.contentDocument
    const win = iframe?.contentWindow
    if (!doc || !win) return

    const doneEl = doc.querySelector("#status[data-state='done']")
    const isDone = !!doneEl
    const scoreEl = doc.getElementById('score')
    const score = scoreEl ? Number.parseFloat(scoreEl.textContent || '0') || 0 : 0
    const canvas = doc.getElementById('game') as HTMLCanvasElement | null
    if (!canvas) return

    const frame = canvas.toDataURL('image/png').split(',')[1] || ''
    const res = await fetch('/api/eval/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame, done: isDone, score }),
    })
    const data = await res.json() as { ok?: boolean; action?: string; complete?: boolean; error?: string }
    if (!res.ok || data.ok === false || data.error) {
      throw new Error(data.error || res.statusText)
    }
    if (data.complete) {
      stopAiLoop(false)
      return
    }
    if (isDone) {
      window.setTimeout(() => {
        reloadGame()
        scheduleEpisodeStart()
      }, 250)
      return
    }
    if (data.action && data.action !== 'noop') {
      applyKeyAction(win, doc, data.action)
    }
  } catch (exc) {
    evalClientError.value = exc instanceof Error ? exc.message : String(exc)
    stopAiLoop(true)
  } finally {
    aiStepInFlight = false
  }
}

function reloadGame() {
  gameReloadToken.value = Date.now()
}

function scheduleEpisodeStart(attempt = 0) {
  window.setTimeout(() => {
    if (!isEvaluating.value && !isTraining.value) return
    if (sendConfiguredKey(props.resetKey || 'Space')) return
    if (attempt < 10) scheduleEpisodeStart(attempt + 1)
  }, attempt === 0 ? 500 : 150)
}

function sendConfiguredKey(rawKey: string): boolean {
  const win = gameFrame.value?.contentWindow
  const doc = gameFrame.value?.contentDocument
  if (!win || !doc || !doc.getElementById('game')) return false
  const { key, code } = normalizeKey(rawKey)
  dispatchKey(win, doc, 'keydown', key, code)
  window.setTimeout(() => dispatchKey(win, doc, 'keyup', key, code), 80)
  return true
}

function applyKeyAction(win: Window, doc: Document, action: string) {
  const keyMap: Record<string, { key: string; code: string }> = {
    ArrowLeft: { key: 'ArrowLeft', code: 'ArrowLeft' },
    ArrowRight: { key: 'ArrowRight', code: 'ArrowRight' },
    Space: { key: ' ', code: 'Space' },
  }
  const keys = action.split('+').map(raw => keyMap[raw] ?? { key: raw, code: raw })
  keys.forEach(({ key, code }) => dispatchKey(win, doc, 'keydown', key, code))
  window.setTimeout(() => {
    keys.slice().reverse().forEach(({ key, code }) => dispatchKey(win, doc, 'keyup', key, code))
  }, 80)
}

function normalizeKey(rawKey: string): { key: string; code: string } {
  if (rawKey === 'Space' || rawKey === ' ') return { key: ' ', code: 'Space' }
  if (rawKey === 'ArrowLeft') return { key: 'ArrowLeft', code: 'ArrowLeft' }
  if (rawKey === 'ArrowRight') return { key: 'ArrowRight', code: 'ArrowRight' }
  if (rawKey.length === 1) return { key: rawKey, code: `Key${rawKey.toUpperCase()}` }
  return { key: rawKey, code: rawKey }
}

function dispatchKey(win: Window, doc: Document, type: 'keydown' | 'keyup', key: string, code: string) {
  const event = new KeyboardEvent(type, { key, code, bubbles: true, cancelable: true })
  win.dispatchEvent(event)
  doc.dispatchEvent(new KeyboardEvent(type, { key, code, bubbles: true, cancelable: true }))
}

function fmtNum(v: unknown): string {
  if (v == null || typeof v !== 'number') return '—'
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}
</script>

<template>
  <div :class="sectionClass" id="gameSection">
    <div class="game-header">
      <span class="gh-title" id="gameTitle">{{ gameTitle }}</span>
      <span class="gh-status" id="gameStatus">{{ gameStatus }}</span>
      <span class="gh-stat" id="gameStats" v-html="gameStats"></span>
      <button @click="toggleGame" :title="showGame ? 'Hide game view' : 'Show game view'" style="background:transparent;border:1px solid var(--border);border-radius:2px;color:var(--muted);padding:2px 8px;font-size:10px;cursor:pointer;font-family:'Outfit',sans-serif;">
        {{ showGame ? 'hide' : 'show' }}
      </button>
    </div>
    <div class="game-body" v-show="showGame">
      <iframe ref="gameFrame" id="gameFrame" :src="gameSrc" scrolling="no" style="overflow:hidden;" title="Game canvas"></iframe>
    </div>
    <img class="train-frame" id="trainFrame" :src="trainFrameSrc" alt="training view" title="Live training screenshot" />
    <div class="game-controls" v-show="showGame">
      <span class="ctrl-label">play</span>
      <button @click="gameAction('left')" title="Send ArrowLeft">&#9664; left</button>
      <button @click="gameAction('space')" title="Send Space">serve</button>
      <button @click="gameAction('right')" title="Send ArrowRight">right &#9654;</button>
      <button @click="gameAction('reset')" title="Send R">reset</button>
    </div>
  </div>
</template>
