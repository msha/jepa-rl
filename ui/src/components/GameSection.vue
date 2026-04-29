<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount, onMounted, nextTick } from 'vue'
import { usePolling } from '../composables/usePolling'
import { useConfigStore } from '../stores/config'
import VDropdown from './VDropdown.vue'
import type { Job, EvalJob } from '../stores/training'

const configStore = useConfigStore()
const selectedConfig = ref('')

const configOptions = computed(() =>
  configStore.configs.map(c => ({ value: c.path, label: c.name }))
)

onMounted(async () => {
  await configStore.loadConfigs()
  selectedConfig.value = configStore.currentPath
})

async function onConfigChange() {
  if (!selectedConfig.value) return
  try {
    await configStore.switchConfig(selectedConfig.value)
    reloadGame()
  } catch (e) {
    console.error(e)
  }
}

const props = defineProps<{
  job: Job | null
  evalJob: EvalJob | null
  resetKey: string
  actionKeys: string[]
  steps: Record<string, unknown>[]
  playerName: string
}>()

const isTraining = computed(() => !!props.job?.running || props.job?.status === 'running' || props.job?.status === 'starting')
const isEvaluating = computed(() => !isTraining.value && (!!props.evalJob?.running || props.evalJob?.status === 'running'))

const showGame = ref(true)
const gameStats = ref('')
const gameFrame = ref<HTMLIFrameElement | null>(null)
const focusOverlay = ref<HTMLDivElement | null>(null)
const gameReloadToken = ref(0)
const evalClientError = ref('')
const gameHighscores = ref<{ score: number; player: string }[]>([])

const emit = defineEmits<{ highscores: [scores: { score: number; player: string }[]] }>()

function onGameMessage(event: MessageEvent) {
  if (event.data?.type !== 'jeparl-scores') return
  gameHighscores.value = event.data.scores || []
  emit('highscores', gameHighscores.value)
}
onMounted(() => {
  window.addEventListener('message', onGameMessage)
})
onBeforeUnmount(() => {
  window.removeEventListener('message', onGameMessage)
})

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

const currentTrainingStep = computed(() => {
  const last = props.steps[props.steps.length - 1]
  return typeof last?.step === 'number' ? last.step : 0
})

const gameStatus = computed(() => {
  if (evalClientError.value) return `eval error: ${evalClientError.value}`
  if (isTraining.value) return `training · step ${currentTrainingStep.value}/${props.job?.requested_steps ?? 0}`
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

const hasArrow = (dir: string) => props.actionKeys.some(k => k.includes(dir))
const showUp = computed(() => hasArrow('ArrowUp'))
const showDown = computed(() => hasArrow('ArrowDown'))
const showLeft = computed(() => hasArrow('ArrowLeft'))
const showRight = computed(() => hasArrow('ArrowRight'))

const isIdle = computed(() => !isTraining.value && !isEvaluating.value)

const heldDirs = ref<string[]>([])
let clickLitTimer: ReturnType<typeof window.setTimeout> | null = null
const gameFocused = ref(false)
const inputMode = ref<'arrows' | 'wasd'>('arrows')
const recentlyActive = ref(false)
let activeTimer: ReturnType<typeof window.setTimeout> | null = null
const spaceNeeded = ref(true)
const prevLives = ref(3)

const litDir = computed(() => {
  const list = heldDirs.value
  return list.length > 0 ? list[list.length - 1] : null
})

function pressDir(dir: string) {
  if (!heldDirs.value.includes(dir)) {
    heldDirs.value = [...heldDirs.value, dir]
  }
}

function releaseDir(dir: string) {
  heldDirs.value = heldDirs.value.filter(d => d !== dir)
}

// For button clicks — auto-release after 200ms
function lightDir(dir: string) {
  pressDir(dir)
  if (clickLitTimer) clearTimeout(clickLitTimer)
  clickLitTimer = setTimeout(() => releaseDir(dir), 200)
}

// Space is a continuous gameplay action (e.g. shoot in asteroids) vs start-only (breakout serve)
const spaceIsContinuous = computed(() =>
  props.actionKeys.some(k => k.includes('Space') && k.includes('+'))
)
const spaceActive = computed(() => spaceNeeded.value || spaceIsContinuous.value)

const gameStarted = ref(false)

function markActive() {
  recentlyActive.value = true
  if (activeTimer) clearTimeout(activeTimer)
  activeTimer = setTimeout(() => { recentlyActive.value = false }, 2000)
}

const keyModeClass = computed(() => {
  if (!isIdle.value || gameStarted.value) {
    return inputMode.value === 'wasd' ? 'mode-wasd' : 'mode-arrows'
  }
  return 'idle-morph'
})

function mapKey(e: KeyboardEvent): { key: string; code: string } | null {
  const k = e.key
  if (k === 'w' || k === 'W') return { key: 'ArrowUp', code: 'ArrowUp' }
  if (k === 'a' || k === 'A') return { key: 'ArrowLeft', code: 'ArrowLeft' }
  if (k === 's' || k === 'S') return { key: 'ArrowDown', code: 'ArrowDown' }
  if (k === 'd' || k === 'D') return { key: 'ArrowRight', code: 'ArrowRight' }
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) return { key: k, code: k }
  if (k === ' ') return { key: ' ', code: 'Space' }
  if (k === 'r' || k === 'R') return { key: 'r', code: 'KeyR' }
  return null
}

function forwardKey(type: 'keydown' | 'keyup', key: string, code: string) {
  const win = gameFrame.value?.contentWindow
  const doc = gameFrame.value?.contentDocument
  if (!win || !doc) return
  dispatchKey(win, doc, type, key, code)
}

function onOverlayKeyDown(e: KeyboardEvent) {
  if (e.key === 'Escape') { gameFocused.value = false; return }

  const k = e.key
  if (['w', 'W', 'a', 'A', 's', 'S', 'd', 'D'].includes(k)) inputMode.value = 'wasd'
  else if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) inputMode.value = 'arrows'

  const mapping = mapKey(e)
  if (!mapping) return
  e.preventDefault()
  forwardKey('keydown', mapping.key, mapping.code)
  markActive()
  gameStarted.value = true

  if (mapping.code === 'ArrowUp') pressDir('up')
  else if (mapping.code === 'ArrowDown') pressDir('down')
  else if (mapping.code === 'ArrowLeft') pressDir('left')
  else if (mapping.code === 'ArrowRight') pressDir('right')
  else if (mapping.code === 'Space') spaceNeeded.value = false
}

function onOverlayKeyUp(e: KeyboardEvent) {
  const mapping = mapKey(e)
  if (!mapping) return
  e.preventDefault()
  forwardKey('keyup', mapping.key, mapping.code)

  if (mapping.code === 'ArrowUp') releaseDir('up')
  else if (mapping.code === 'ArrowDown') releaseDir('down')
  else if (mapping.code === 'ArrowLeft') releaseDir('left')
  else if (mapping.code === 'ArrowRight') releaseDir('right')
}

function focusGameOverlay(e?: MouseEvent) {
  if (e) {
    const target = e.target as HTMLElement
    if (target.tagName === 'SELECT' || target.tagName === 'INPUT' || target.closest('.vdd')) return
  }
  gameFocused.value = true
  nextTick(() => focusOverlay.value?.focus())
}

const sectionClass = computed(() => ({
  'game-section': true,
  'game-focused': gameFocused.value,
  training: isTraining.value,
  evaluating: isEvaluating.value,
}))

const trainFrameSrc = ref('')
const gameSrc = computed(() => {
  const ts = gameReloadToken.value ? `&ts=${gameReloadToken.value}` : ''
  if (isTraining.value) {
    const runName = encodeURIComponent(props.job?.run_name || '')
    return `/game?embed&run_name=${runName}${ts}`
  }
  if (isEvaluating.value) {
    const runName = encodeURIComponent(props.evalJob?.run_name || '')
    const ckpt = encodeURIComponent(props.evalJob?.checkpoint_name || '')
    return `/game?embed&run_name=${runName}&checkpoint=${ckpt}${ts}`
  }
  const name = encodeURIComponent(props.playerName || 'HUMAN')
  return `/game?embed&player_name=${name}${ts}`
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
      const newLives = parseInt(lives.textContent || '3', 10)
      if (newLives < prevLives.value) spaceNeeded.value = true
      prevLives.value = newLives
    }
    const doneEl = doc.querySelector("#status[data-state='done']")
    if (doneEl && gameStarted.value) {
      gameStarted.value = false
      spaceNeeded.value = false
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
  if (activeTimer) clearTimeout(activeTimer)
  if (clickLitTimer) clearTimeout(clickLitTimer)
})

function toggleGame() {
  showGame.value = !showGame.value
}

function gameAction(action: string) {
  if (action === 'space') { spaceNeeded.value = false; gameStarted.value = true }
  const win = gameFrame.value?.contentWindow
  const doc = gameFrame.value?.contentDocument
  if (!win || !doc) return
  const keyMap: Record<string, { key: string; code: string }> = {
    up: { key: 'ArrowUp', code: 'ArrowUp' },
    left: { key: 'ArrowLeft', code: 'ArrowLeft' },
    down: { key: 'ArrowDown', code: 'ArrowDown' },
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
  spaceNeeded.value = true
  prevLives.value = 3
  gameStarted.value = false
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
  <div :class="sectionClass" id="gameSection" @click="focusGameOverlay">

    <div class="game-header">
      <VDropdown
        v-model="selectedConfig"
        :options="configOptions"
        @change="onConfigChange"
        title="Available game configs in configs/games/"
        compact
      />
      <span class="gh-title" id="gameTitle">{{ gameTitle }}</span>
      <span class="gh-status" id="gameStatus">{{ gameStatus }}</span>
      <span class="gh-stat" id="gameStats" v-html="gameStats"></span>
      <button @click="toggleGame" :title="showGame ? 'Hide game view' : 'Show game view'" class="btn-tiny">
        {{ showGame ? 'hide' : 'show' }}
      </button>
    </div>

    <div class="game-body" v-show="showGame">
      <iframe ref="gameFrame" id="gameFrame" :src="gameSrc" scrolling="no" style="overflow:hidden;" title="Game canvas"></iframe>
      <div v-if="!gameFocused" class="game-overlay game-overlay-click">click to play</div>
    </div>

    <img class="train-frame" id="trainFrame" :src="trainFrameSrc" alt="training view" title="Live training screenshot" />

    <div class="game-controls" v-show="showGame" :class="keyModeClass">
      <button @click="gameAction('space')" class="ctrl-act" :class="{ 'ctrl-flicker': spaceNeeded && !gameFocused, 'ctrl-dir-unused': !spaceActive }" title="Space">space</button>
      <div class="ctrl-dpad">
        <div class="ctrl-dpad-row">
          <button @click="gameAction('up'); lightDir('up')" class="ctrl-dir dir-up" :class="{ 'ctrl-dir-lit': litDir === 'up', 'ctrl-dir-unused': !showUp }" title="ArrowUp">
            <span class="lbl-arrow">&#9650;</span><span class="lbl-wasd">W</span>
          </button>
        </div>
        <div class="ctrl-dpad-row">
          <button @click="gameAction('left'); lightDir('left')" class="ctrl-dir dir-left" :class="{ 'ctrl-dir-lit': litDir === 'left', 'ctrl-dir-unused': !showLeft }" title="ArrowLeft">
            <span class="lbl-arrow">&#9664;</span><span class="lbl-wasd">A</span>
          </button>
          <button @click="gameAction('down'); lightDir('down')" class="ctrl-dir dir-down" :class="{ 'ctrl-dir-lit': litDir === 'down', 'ctrl-dir-unused': !showDown }" title="ArrowDown">
            <span class="lbl-arrow">&#9660;</span><span class="lbl-wasd">S</span>
          </button>
          <button @click="gameAction('right'); lightDir('right')" class="ctrl-dir dir-right" :class="{ 'ctrl-dir-lit': litDir === 'right', 'ctrl-dir-unused': !showRight }" title="ArrowRight">
            <span class="lbl-arrow">&#9654;</span><span class="lbl-wasd">D</span>
          </button>
        </div>
      </div>
      <div class="ctrl-actions">
        <button @click="gameAction('reset')" class="ctrl-act ctrl-act-secondary" title="R">reset</button>
      </div>
    </div>

    <!-- Invisible focus overlay — receives keyboard events, pointer-events:none so clicks pass through -->
    <div v-if="gameFocused" ref="focusOverlay" class="game-focus-overlay" tabindex="0"
         @keydown="onOverlayKeyDown" @keyup="onOverlayKeyUp"></div>
  </div>
</template>

<style scoped>
.game-focused .game-body {
  border-radius: 2px;
  animation: crt-glow 4s ease-in-out infinite, crt-flicker 0.07s infinite;
}
.game-body {
  position: relative;
}
@keyframes crt-glow {
  0%, 100% {
    box-shadow:
      inset 0 0 40px rgba(100, 168, 255, 0.06),
      inset 0 0 80px rgba(100, 168, 255, 0.03),
      0 0 6px rgba(100, 168, 255, 0.15),
      0 0 16px rgba(100, 168, 255, 0.08);
  }
  50% {
    box-shadow:
      inset 0 0 50px rgba(100, 168, 255, 0.08),
      inset 0 0 100px rgba(100, 168, 255, 0.04),
      0 0 8px rgba(100, 168, 255, 0.2),
      0 0 22px rgba(100, 168, 255, 0.1);
  }
}
@keyframes crt-flicker {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.993; }
}
.game-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1;
  outline: none;
}
.game-overlay-click {
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(7, 9, 13, 0.45);
  color: rgba(244, 246, 251, 0.5);
  font-size: 13px;
  font-family: 'Outfit', sans-serif;
  cursor: pointer;
  transition: background 0.15s;
}
.game-overlay-click:hover {
  background: rgba(7, 9, 13, 0.25);
  color: rgba(244, 246, 251, 0.7);
}
.game-focus-overlay {
  position: absolute;
  inset: 0;
  z-index: 10;
  outline: none;
  pointer-events: none;
}

/* Key label morphing */
.ctrl-dir {
  position: relative;
  overflow: hidden;
}
.lbl-arrow, .lbl-wasd {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}

/* Playing: instant opacity switch */
.mode-arrows .lbl-arrow { opacity: 1; transform: translate(-50%, -50%); transition: opacity 0.15s; }
.mode-arrows .lbl-wasd { opacity: 0; transform: translate(-50%, -50%); transition: opacity 0.15s; }
.mode-wasd .lbl-arrow { opacity: 0; transform: translate(-50%, -50%); transition: opacity 0.15s; }
.mode-wasd .lbl-wasd { opacity: 1; transform: translate(-50%, -50%); transition: opacity 0.15s; }

/* Idle: slide-left cascade (7s cycle = 3s arrows + 0.5s slide + 3s WASD + 0.5s slide)
   Cascade order: D(right)→S(down)→W(up)→A(left) with stagger delays */
@keyframes morph-out {
  0%, 42.86% { transform: translate(-50%, -50%); opacity: 1; }
  50% { transform: translate(-50%, -50%) translateX(-120%); opacity: 0; }
  92.86% { transform: translate(-50%, -50%) translateX(120%); opacity: 0; }
  100% { transform: translate(-50%, -50%); opacity: 1; }
}
@keyframes morph-in {
  0%, 42.86% { transform: translate(-50%, -50%) translateX(120%); opacity: 0; }
  50% { transform: translate(-50%, -50%); opacity: 1; }
  92.86% { transform: translate(-50%, -50%); opacity: 1; }
  100% { transform: translate(-50%, -50%) translateX(-120%); opacity: 0; }
}

.idle-morph .lbl-arrow { animation: morph-out 7s ease infinite; }
.idle-morph .lbl-wasd { animation: morph-in 7s ease infinite; }

/* Cascade: right=0ms, down=100ms, up=150ms, left=200ms */
.idle-morph .dir-down .lbl-arrow { animation-delay: 0.1s; }
.idle-morph .dir-down .lbl-wasd  { animation-delay: 0.1s; }
.idle-morph .dir-up .lbl-arrow { animation-delay: 0.15s; }
.idle-morph .dir-up .lbl-wasd  { animation-delay: 0.15s; }
.idle-morph .dir-left .lbl-arrow { animation-delay: 0.2s; }
.idle-morph .dir-left .lbl-wasd  { animation-delay: 0.2s; }

.ctrl-dir-unused {
  opacity: 0.3 !important;
  cursor: default !important;
  pointer-events: none;
}
</style>
