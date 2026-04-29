<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useTrainingStore } from '../stores/training'
import { useRunsStore } from '../stores/runs'
import { api, getJson } from '../api/client'
import type { Job, EvalJob } from '../stores/training'

const props = defineProps<{
  job: Job | null
  evalJob: EvalJob | null
  summary: Record<string, unknown>
  latestStep: Record<string, unknown>
}>()

const training = useTrainingStore()
const runs = useRunsStore()

const targetRunName = ref('')
const additionalSteps = ref(500)
const warmup = ref(50)
const batch = ref('')
const logEvery = ref(5)
const lr = ref('')
const headed = ref(false)
const jepaCheckpoint = ref('')
const selectedCheckpoint = ref('')
const controlError = ref('')

const currentAlgorithm = computed(() => {
  const agentGroup = training.configDetail.find(g => g.title === 'agent')
  if (!agentGroup) return 'linear_q'
  const field = agentGroup.fields.find(f => f[0] === 'algorithm')
  return field ? String(field[1]) : 'linear_q'
})
const isFrozenJepa = computed(() => currentAlgorithm.value === 'frozen_jepa_dqn')
watch(isFrozenJepa, frozen => { if (!frozen && activeStep.value === 2) activeStep.value = 3 })

function onNewRun(e: Event) {
  const detail = (e as CustomEvent).detail
  if (detail?.name) targetRunName.value = detail.name
}
onMounted(() => window.addEventListener('new-run', onNewRun))
onUnmounted(() => window.removeEventListener('new-run', onNewRun))

const now = ref(Math.floor(Date.now() / 1000))
let clock: ReturnType<typeof setInterval> | null = null
onMounted(() => { clock = setInterval(() => { now.value = Math.floor(Date.now() / 1000) }, 1000) })
onUnmounted(() => { if (clock) clearInterval(clock) })

const busy = computed(() => training.isTraining || training.isEvaluating)
const isTraining = computed(() => !!props.job?.running || props.job?.status === 'running' || props.job?.status === 'starting')
const isEvaluating = computed(() => !isTraining.value && (!!props.evalJob?.running || props.evalJob?.status === 'running'))

const dotClass = computed(() => {
  if (isTraining.value) return 'tc-dot running'
  if (isEvaluating.value) return 'tc-dot evaluating'
  if (props.job?.status === 'error' || props.evalJob?.status === 'error') return 'tc-dot error'
  if (props.job?.status === 'completed') return 'tc-dot stopped'
  return 'tc-dot idle'
})

const statusText = computed(() => {
  if (isTraining.value) return 'training'
  if (isEvaluating.value) return 'evaluating'
  if (props.job?.status === 'completed') return 'completed'
  if (props.job?.status === 'stopped') return 'stopped'
  if (props.job?.status === 'error') return 'error'
  if (props.evalJob?.status === 'error') return 'eval error'
  return 'idle'
})

const statusDetail = computed(() => {
  const s = props.summary || {}
  const parts: string[] = []
  if (s.algorithm) {
    const display: Record<string, string> = {
      pixel_dqn: 'dqn', PixelDQN: 'dqn',
      frozen_jepa_dqn: 'frozen jepa dqn', FrozenJEPADQN: 'frozen jepa dqn',
      linear_q: 'linear q', LinearQ: 'linear q',
    }
    parts.push(display[String(s.algorithm)] || String(s.algorithm))
  }
  if (s.episodes) parts.push(`${s.episodes} eps`)
  if (s.best_score != null) parts.push('best:' + fmtNum(s.best_score))
  if (s.steps) parts.push(fmtNum(s.steps) + ' steps')
  return parts.join(' · ')
})

const startTs = computed(() => {
  if (!props.job?.started_at) return null
  return props.job.started_at > 1e12 ? props.job.started_at / 1000 : props.job.started_at
})

const elapsed = computed(() => {
  if (!isTraining.value || startTs.value == null) return null
  return Math.max(0, now.value - Math.floor(startTs.value))
})

const currentStep = computed(() => Number(props.latestStep?.step ?? props.summary?.steps ?? 0))

const eta = computed(() => {
  if (!isTraining.value || startTs.value == null || !props.job?.requested_steps) return null
  if (currentStep.value <= 0) return null
  const elapsedSec = Math.max(1, now.value - Math.floor(startTs.value))
  const rate = currentStep.value / elapsedSec
  const remaining = props.job.requested_steps - currentStep.value
  if (remaining <= 0) return null
  return remaining / rate
})

const progress = computed(() => {
  if (!isTraining.value || !props.job?.requested_steps) return null
  return Math.min(1, currentStep.value / props.job.requested_steps)
})

function fmtDuration(sec: number | null): string {
  if (sec == null || sec < 0) return ''
  sec = Math.floor(sec)
  if (sec < 60) return `${sec}s`
  const m = Math.floor(sec / 60)
  const s = sec % 60
  if (m < 60) return `${m}m ${s}s`
  const hr = Math.floor(m / 60)
  return `${hr}h ${m % 60}m`
}

function fmtNum(v: unknown): string {
  if (v == null || typeof v !== 'number') return '—'
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

function fmtRun(r: { name: string; experiment_name?: string; algorithm?: string; best_score?: number; steps?: number }): string {
  const label = r.experiment_name || r.name
  const parts = [label]
  if (r.algorithm) parts.push(r.algorithm)
  if (r.best_score != null) parts.push('best:' + fmtNum(r.best_score))
  if (r.steps != null) parts.push(r.steps + ' steps')
  return parts.join(' · ')
}

function onRunChange() {
  runs.loadRunDetail(runs.selectedRun)
}

async function deleteSelectedRun() {
  if (!runs.selectedRun) return
  if (!confirm(`Delete run "${runs.selectedRun}" and all its data?`)) return
  try { await runs.deleteRun(runs.selectedRun) }
  catch (e) { controlError.value = e instanceof Error ? e.message : String(e) }
}

// New-run popover
const showNewRun = ref(false)
const newRunName = ref('')
const newRunPopoverStyle = ref<Record<string, string>>({})

function openNewRun(e: MouseEvent) {
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  newRunName.value = `run_${ts}`
  newRunPopoverStyle.value = anchorBelow(e.currentTarget as HTMLElement, 200)
  showNewRun.value = true
}

function confirmNewRun() {
  const name = newRunName.value.trim()
  if (!name) return
  runs.selectedRun = ''
  runs.checkpoints = []
  runs.runDir = ''
  runs.runConfigDetail = null
  window.dispatchEvent(new CustomEvent('new-run', { detail: { name } }))
  targetRunName.value = name
  showNewRun.value = false
}

const availableCheckpoints = computed(() => {
  if (runs.checkpoints.length) return runs.checkpoints
  return training.runDir?.checkpoints ?? []
})
const evalRunDir = computed(() => runs.runDir || training.runDir?.dir || '')
const hasCheckpoint = computed(() => availableCheckpoints.value.length > 0)

function checkpointStep(): number {
  if (!selectedCheckpoint.value) return 0
  const m = selectedCheckpoint.value.match(/(\d+)/)
  return m ? parseInt(m[1], 10) : 0
}

watch(availableCheckpoints, checkpoints => {
  if (!checkpoints.length) { selectedCheckpoint.value = ''; return }
  const files = checkpoints.map(c => c.file)
  if (!selectedCheckpoint.value || !files.includes(selectedCheckpoint.value)) {
    selectedCheckpoint.value = files[files.length - 1]
  }
}, { immediate: true })

onMounted(async () => {
  try {
    const defaults = await getJson<{
      default_steps: number; learning_starts: number; batch_size: number | null; dashboard_every: number
    }>('/api/defaults')
    additionalSteps.value = defaults.default_steps || 500
    warmup.value = defaults.learning_starts ?? 50
    batch.value = defaults.batch_size != null ? String(defaults.batch_size) : ''
    logEvery.value = defaults.dashboard_every || 5
  } catch { /* use defaults */ }
})

async function startTraining() {
  controlError.value = ''
  const experiment = targetRunName.value || runs.selectedRun || ''
  if (!experiment) { controlError.value = 'select or create a run first'; return }
  const ckptStep = checkpointStep()
  const totalSteps = ckptStep + Number(additionalSteps.value)
  const payload: Record<string, unknown> = {
    experiment, steps: totalSteps,
    learning_starts: Number(warmup.value),
    batch_size: batch.value || null,
    dashboard_every: Number(logEvery.value) || 5,
    headed: headed.value,
  }
  if (lr.value) payload.lr = Number(lr.value)
  if (isFrozenJepa.value && jepaCheckpoint.value) payload.jepa_checkpoint = jepaCheckpoint.value
  try {
    await api('/api/train/start', payload)
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
  setTimeout(() => runs.loadRuns(), 500)
}

async function stopTraining() {
  controlError.value = ''
  try {
    if (training.isEvaluating) await api('/api/eval/stop')
    else await api('/api/train/stop')
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
}

async function watchAiPlay() {
  controlError.value = ''
  if (!selectedCheckpoint.value) { controlError.value = 'select a checkpoint first'; return }
  const payload: Record<string, unknown> = { episodes: 3, checkpoint: selectedCheckpoint.value }
  if (evalRunDir.value) payload.run_dir = evalRunDir.value
  if (isFrozenJepa.value && jepaCheckpoint.value) payload.jepa_checkpoint = jepaCheckpoint.value
  try {
    await api('/api/eval/start', payload)
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
}

const collectEpisodes = ref(5)
const collectMaxSteps = ref(200)
const collectExperiment = ref('')
const collectError = ref('')

async function startCollect() {
  collectError.value = ''
  try {
    await training.startCollect({
      experiment: collectExperiment.value || undefined,
      episodes: collectEpisodes.value,
      max_steps: collectMaxSteps.value,
      headed: headed.value,
    })
  } catch (e) { collectError.value = e instanceof Error ? e.message : String(e) }
}

async function stopCollect() {
  try { await training.stopCollect() }
  catch (e) { collectError.value = e instanceof Error ? e.message : String(e) }
}

const worldSteps = ref(1000)
const worldCollectSteps = ref('')
const worldBatch = ref('')
const worldLr = ref('')
const worldDashEvery = ref(25)
const worldError = ref('')

async function startWorldTraining() {
  worldError.value = ''
  try {
    await training.startWorldTraining({
      experiment: (targetRunName.value || runs.selectedRun || undefined) && `${targetRunName.value || runs.selectedRun}_world`,
      steps: worldSteps.value,
      collect_steps: worldCollectSteps.value ? Number(worldCollectSteps.value) : undefined,
      batch_size: worldBatch.value ? Number(worldBatch.value) : undefined,
      lr: worldLr.value ? Number(worldLr.value) : undefined,
      dashboard_every: worldDashEvery.value,
      headed: headed.value,
    })
  } catch (e) { worldError.value = e instanceof Error ? e.message : String(e) }
}

async function stopWorldTraining() {
  try { await training.stopWorldTraining() }
  catch (e) { worldError.value = e instanceof Error ? e.message : String(e) }
}

const smokeSteps = ref(2000)
const smokeLr = ref(0.03)
const smokeSeed = ref(0)
const smokeResult = ref<{ ok: boolean; passed?: boolean; improvement?: number; error?: string } | null>(null)
const smokeRunning = ref(false)
const validateResult = ref<{ ok: boolean; game?: string; algorithm?: string; error?: string } | null>(null)

async function runValidate() {
  validateResult.value = null
  validateResult.value = await training.validateConfig()
}

async function runSmoke() {
  smokeResult.value = null
  smokeRunning.value = true
  try {
    smokeResult.value = await training.runMlSmoke({ steps: smokeSteps.value, lr: smokeLr.value, seed: smokeSeed.value })
  } finally { smokeRunning.value = false }
}

// Field edit popover
const editingField = ref<string | null>(null)
const editValue = ref('')
const editPopoverStyle = ref<Record<string, string>>({})

function displayVal(key: string): string {
  switch (key) {
    case 'collectEpisodes': return String(collectEpisodes.value)
    case 'collectMaxSteps': return String(collectMaxSteps.value)
    case 'collectExperiment': return collectExperiment.value
    case 'worldSteps': return String(worldSteps.value)
    case 'worldBatch': return worldBatch.value
    case 'worldLr': return worldLr.value
    case 'worldCollectSteps': return worldCollectSteps.value
    case 'worldDashEvery': return String(worldDashEvery.value)
    case 'warmup': return String(warmup.value)
    case 'additionalSteps': return String(additionalSteps.value)
    case 'batch': return batch.value
    case 'lr': return lr.value
    case 'logEvery': return String(logEvery.value)
    case 'jepaCheckpoint': return jepaCheckpoint.value
    default: return ''
  }
}

function openFieldEdit(key: string, event: MouseEvent) {
  editingField.value = key
  editValue.value = displayVal(key)
  editPopoverStyle.value = anchorBelow(event.currentTarget as HTMLElement, 200)
}

function saveFieldEdit() {
  if (!editingField.value) return
  const key = editingField.value
  const val = editValue.value
  const n = Number(val)
  switch (key) {
    case 'collectEpisodes': collectEpisodes.value = n; break
    case 'collectMaxSteps': collectMaxSteps.value = n; break
    case 'collectExperiment': collectExperiment.value = val; break
    case 'worldSteps': worldSteps.value = n; break
    case 'worldBatch': worldBatch.value = val; break
    case 'worldLr': worldLr.value = val; break
    case 'worldCollectSteps': worldCollectSteps.value = val; break
    case 'worldDashEvery': worldDashEvery.value = n; break
    case 'warmup': warmup.value = n; break
    case 'additionalSteps': additionalSteps.value = n; break
    case 'batch': batch.value = val; break
    case 'lr': lr.value = val; break
    case 'logEvery': logEvery.value = n; break
    case 'jepaCheckpoint': jepaCheckpoint.value = val; break
  }
  editingField.value = null
}

function closeFieldEdit() {
  editingField.value = null
}

// Pipeline
const activeStep = ref(3)

type StepStatus = 'pending' | 'ready' | 'done' | 'running' | 'error' | 'skipped'

const stepStatus = computed((): StepStatus[] => {
  const s0: StepStatus = validateResult.value?.ok ? 'done' : validateResult.value ? 'error' : 'ready'
  const s1: StepStatus = training.isCollecting ? 'running'
    : training.collectJob?.status === 'completed' ? 'done'
    : training.collectJob?.status === 'error' ? 'error' : 'ready'
  const s2: StepStatus = !isFrozenJepa.value ? 'skipped'
    : training.isWorldTraining ? 'running'
    : training.worldJob?.status === 'completed' ? 'done'
    : training.worldJob?.status === 'error' ? 'error' : 'ready'
  const s3: StepStatus = isTraining.value ? 'running'
    : props.job?.status === 'completed' ? 'done'
    : props.job?.status === 'error' ? 'error'
    : (targetRunName.value || runs.selectedRun) ? 'ready' : 'pending'
  return [s0, s1, s2, s3]
})

const STEP_LABELS = ['check', 'data', 'jepa', 'train']
const STEP_TITLES = ['System Checks', 'Collect Data', 'Pretrain JEPA', 'Train RL Agent']

const visibleSteps = computed(() => isFrozenJepa.value ? [0, 1, 2, 3] : [0, 1, 3])

function nextStep(current: number): number {
  const idx = visibleSteps.value.indexOf(current)
  return idx >= 0 && idx < visibleSteps.value.length - 1 ? visibleSteps.value[idx + 1] : current
}
function prevStep(current: number): number {
  const idx = visibleSteps.value.indexOf(current)
  return idx > 0 ? visibleSteps.value[idx - 1] : current
}

function nodeIcon(status: StepStatus, idx: number): string {
  if (status === 'done') return '✓'
  if (status === 'error') return '✗'
  if (status === 'skipped') return '–'
  if (status === 'running') return '●'
  return String(idx + 1)
}

// Shared popover positioning helper
function anchorBelow(el: HTMLElement, minWidth = 180): Record<string, string> {
  const rect = el.getBoundingClientRect()
  const w = Math.max(minWidth, rect.width)
  let top = rect.bottom + 4
  let left = rect.left
  if (top + 110 > window.innerHeight - 8) top = rect.top - 110 - 4
  if (left + w > window.innerWidth - 8) left = rect.right - w
  if (left < 8) left = 8
  return { top: `${top}px`, left: `${left}px`, width: `${w}px` }
}
</script>

<template>
  <div class="pipeline" id="controlGroup" :data-run-dir="runs.runDir">

    <!-- Run selector row -->
    <div class="pipe-run-row">
      <select
        v-model="runs.selectedRun"
        @change="onRunChange"
        class="pipe-run-select"
        title="Select a training run"
      >
        <option value="">select run...</option>
        <option v-for="r in runs.runs" :key="r.name" :value="r.name">{{ fmtRun(r) }}</option>
      </select>
      <button @click="openNewRun" class="btn-tiny">+ new</button>
      <label class="pipe-smoke-lbl" title="Include smoke test runs">
        <input type="checkbox" v-model="runs.showSmoke" @change="runs.loadRuns()"> smoke
      </label>
      <button v-if="runs.selectedRun" @click="deleteSelectedRun" class="btn-danger-tiny">del</button>
    </div>

    <!-- Status row -->
    <div class="pipe-status">
      <span :class="dotClass"></span>
      <span class="pipe-st-label">{{ statusText }}</span>
      <span class="pipe-st-detail">{{ statusDetail }}</span>
      <template v-if="isTraining && elapsed != null">
        <span class="pipe-st-time">{{ fmtDuration(elapsed) }}</span>
        <span v-if="eta != null" class="pipe-st-eta">~{{ fmtDuration(eta) }}</span>
      </template>
      <button v-if="busy" @click="stopTraining" class="btn-danger-tiny">stop</button>
    </div>
    <div v-if="isTraining && progress != null" class="pipe-prog">
      <div class="pipe-prog-fill" :style="{ width: (progress * 100) + '%' }"></div>
    </div>
    <div class="tc-error pipe-err" v-if="controlError">{{ controlError }}</div>

    <!-- Step track -->
    <div class="pipe-track">
      <template v-for="(si, vi) in visibleSteps" :key="si">
        <button
          class="pipe-node"
          :class="['ns-' + stepStatus[si], { 'ns-active': activeStep === si }]"
          @click="activeStep = si"
          :title="STEP_TITLES[si]"
        >
          <span class="pipe-node-ic">{{ nodeIcon(stepStatus[si], si) }}</span>
          <span class="pipe-node-lb">{{ STEP_LABELS[si] }}</span>
        </button>
        <div v-if="vi < visibleSteps.length - 1" class="pipe-conn" :class="{ 'pipe-conn-done': stepStatus[si] === 'done', 'pipe-conn-ready': stepStatus[si] === 'ready' || stepStatus[si] === 'running' }"></div>
      </template>
    </div>

    <!-- Step 0: System Checks -->
    <div v-show="activeStep === 0" class="settings-table pipe-panel">
      <span class="sk" title="Validate config against schema">validate</span>
      <div class="sv" style="display:flex;align-items:center;gap:6px;">
        <span v-if="validateResult" :class="validateResult.ok ? 'tc-badge-ok' : 'tc-badge-err'" style="font-size:9px;">
          {{ validateResult.ok ? `✓ ${validateResult.game}` : `✗ ${validateResult.error}` }}
        </span>
        <button @click="runValidate" class="btn-tiny" style="margin-left:auto;">run</button>
      </div>

      <span class="sk" title="Synthetic gradient check">ml smoke</span>
      <div class="sv" style="display:flex;align-items:center;gap:5px;flex-wrap:wrap;">
        <label title="Steps" class="pipe-lbl">st<input v-model.number="smokeSteps" type="number" min="1" class="pipe-ii" style="width:36px;"></label>
        <label title="Learning rate" class="pipe-lbl">lr<input v-model.number="smokeLr" type="number" step="any" class="pipe-ii" style="width:36px;"></label>
        <label title="Seed" class="pipe-lbl">sd<input v-model.number="smokeSeed" type="number" min="0" class="pipe-ii" style="width:28px;"></label>
        <button @click="runSmoke" class="btn-tiny" :disabled="smokeRunning" style="margin-left:auto;">{{ smokeRunning ? '…' : 'run' }}</button>
      </div>

      <template v-if="smokeResult">
        <span class="sk"></span>
        <span :class="smokeResult.passed ? 'tc-badge-ok' : 'tc-badge-err'" class="sv" style="font-size:9px;">
          {{ smokeResult.ok ? (smokeResult.passed ? '✓ passed' : '✗ failed') + ' · Δ' + fmtNum(smokeResult.improvement) : '✗ ' + smokeResult.error }}
        </span>
      </template>

      <span class="sk"></span>
      <div class="sv pipe-nav">
        <button @click="activeStep = nextStep(0)" class="btn-tiny pipe-next">next →</button>
      </div>
    </div>

    <!-- Step 1: Collect Data -->
    <div v-show="activeStep === 1" class="settings-table pipe-panel">
      <span class="sk" title="Random episodes">episodes</span>
      <div class="sv clickable-field" @click="openFieldEdit('collectEpisodes', $event)">{{ collectEpisodes }}</div>

      <span class="sk" title="Max steps per episode">max steps</span>
      <div class="sv clickable-field" @click="openFieldEdit('collectMaxSteps', $event)">{{ collectMaxSteps }}</div>

      <span class="sk" title="Experiment name">name</span>
      <div class="sv clickable-field" @click="openFieldEdit('collectExperiment', $event)">
        <span v-if="collectExperiment">{{ collectExperiment }}</span>
        <span v-else class="cf-muted">auto</span>
      </div>

      <template v-if="training.isCollecting && training.collectJob">
        <span class="sk">progress</span>
        <span class="sv" style="color:var(--accent);">{{ training.collectJob.episodes_done }}/{{ training.collectJob.episodes_target }}</span>
      </template>
      <template v-if="training.isCollecting && training.collectJob?.mean_score">
        <span class="sk">mean score</span>
        <span class="sv tc-badge-ok">{{ fmtNum(training.collectJob.mean_score) }}</span>
      </template>

      <div v-if="collectError" class="tc-error" style="grid-column:1/-1;font-size:9px;">{{ collectError }}</div>

      <span class="sk"></span>
      <div class="sv pipe-nav">
        <button @click="activeStep = prevStep(1)" class="btn-tiny">← back</button>
        <button v-if="!training.isCollecting" @click="startCollect" class="btn-tiny" :disabled="busy">collect</button>
        <button v-if="training.isCollecting" @click="stopCollect" class="btn-tiny" style="color:var(--red);">stop</button>
        <button @click="activeStep = nextStep(1)" class="btn-tiny pipe-next">next →</button>
      </div>
    </div>

    <!-- Step 2: Pretrain JEPA -->
    <div v-show="activeStep === 2" class="settings-table pipe-panel">
      <template v-if="!isFrozenJepa">
        <span class="sk"></span>
        <span class="sv" style="color:var(--muted);font-size:9px;font-style:italic;">skip — not needed for {{ currentAlgorithm }}</span>
      </template>
      <template v-else>
        <span class="sk" title="Gradient steps">steps</span>
        <div class="sv clickable-field" @click="openFieldEdit('worldSteps', $event)">{{ worldSteps }}</div>

        <span class="sk" title="Batch size">batch</span>
        <div class="sv clickable-field" @click="openFieldEdit('worldBatch', $event)">
          <span v-if="worldBatch">{{ worldBatch }}</span>
          <span v-else class="cf-muted">auto</span>
        </div>

        <span class="sk" title="Learning rate">lr</span>
        <div class="sv clickable-field" @click="openFieldEdit('worldLr', $event)">
          <span v-if="worldLr">{{ worldLr }}</span>
          <span v-else class="cf-muted">auto</span>
        </div>

        <span class="sk" title="Pre-collect browser steps">pre-collect</span>
        <div class="sv clickable-field" @click="openFieldEdit('worldCollectSteps', $event)">
          <span v-if="worldCollectSteps">{{ worldCollectSteps }}</span>
          <span v-else class="cf-muted">auto</span>
        </div>

        <span class="sk" title="Log interval">log every</span>
        <div class="sv clickable-field" @click="openFieldEdit('worldDashEvery', $event)">{{ worldDashEvery }}</div>

        <template v-if="training.isWorldTraining && training.worldJob">
          <span class="sk">run</span>
          <span class="sv" style="color:var(--accent);">{{ training.worldJob.run_name }}</span>
        </template>
        <div v-if="worldError" class="tc-error" style="grid-column:1/-1;font-size:9px;">{{ worldError }}</div>
      </template>

      <span class="sk"></span>
      <div class="sv pipe-nav">
        <button @click="activeStep = prevStep(2)" class="btn-tiny">← back</button>
        <template v-if="isFrozenJepa">
          <button v-if="!training.isWorldTraining" @click="startWorldTraining" class="btn-tiny" :disabled="busy">train</button>
          <button v-if="training.isWorldTraining" @click="stopWorldTraining" class="btn-tiny" style="color:var(--red);">stop</button>
        </template>
        <button @click="activeStep = nextStep(2)" class="btn-tiny pipe-next">next →</button>
      </div>
    </div>

    <!-- Step 3: Train RL Agent -->
    <div v-show="activeStep === 3" class="settings-table pipe-panel">
      <template v-if="isFrozenJepa">
        <span class="sk" title="JEPA encoder checkpoint">jepa ckpt</span>
        <div class="sv clickable-field" @click="openFieldEdit('jepaCheckpoint', $event)">
          <span v-if="jepaCheckpoint">{{ jepaCheckpoint }}</span>
          <span v-else class="cf-muted">none</span>
        </div>
      </template>

      <template v-if="hasCheckpoint">
        <span class="sk" title="Resume from checkpoint">from</span>
        <select v-model="selectedCheckpoint" class="sv pipe-ii" style="width:100%;">
          <option v-for="c in availableCheckpoints" :key="c.file" :value="c.file">{{ c.label }}</option>
        </select>
      </template>

      <span class="sk" title="Steps before model updates">warmup</span>
      <div class="sv clickable-field" @click="openFieldEdit('warmup', $event)">{{ warmup }}</div>

      <span class="sk" title="Additional training steps">+ steps</span>
      <div class="sv clickable-field" @click="openFieldEdit('additionalSteps', $event)" style="display:flex;align-items:center;gap:4px;">
        {{ additionalSteps }}
        <span v-if="hasCheckpoint && checkpointStep() > 0" style="color:var(--muted);font-size:9px;">→ {{ checkpointStep() + additionalSteps }}</span>
      </div>

      <span class="sk" title="Minibatch size">batch</span>
      <div class="sv clickable-field" @click="openFieldEdit('batch', $event)">
        <span v-if="batch">{{ batch }}</span>
        <span v-else class="cf-muted">auto</span>
      </div>

      <span class="sk" title="Learning rate override">lr</span>
      <div class="sv clickable-field" @click="openFieldEdit('lr', $event)">
        <span v-if="lr">{{ lr }}</span>
        <span v-else class="cf-muted">auto</span>
      </div>

      <span class="sk" title="Log every N steps">log every</span>
      <div class="sv clickable-field" @click="openFieldEdit('logEvery', $event)">{{ logEvery }}</div>

      <span class="sk" title="Show browser window">headed</span>
      <div class="sv" style="display:flex;align-items:center;">
        <input type="checkbox" v-model="headed" :disabled="busy" style="margin:0;accent-color:var(--accent);">
      </div>

      <span class="sk"></span>
      <div class="sv pipe-nav">
        <button @click="activeStep = prevStep(3)" class="btn-tiny">← back</button>
        <button @click="watchAiPlay" v-if="!busy && hasCheckpoint" class="btn-accent-tiny">watch</button>
        <button @click="startTraining" v-if="!busy" class="btn-accent-tiny">train</button>
        <button @click="stopTraining" v-if="busy" class="btn-danger-tiny">stop</button>
      </div>
    </div>

    <!-- Field edit popover -->
    <div v-if="editingField" class="popover-backdrop" @click="closeFieldEdit"></div>
    <div v-if="editingField" class="popover" :style="editPopoverStyle">
      <div class="popover-label">{{ editingField }}</div>
      <input v-model="editValue" type="text" class="popover-input" @keydown.enter="saveFieldEdit" @keydown.escape="closeFieldEdit" autofocus>
      <div class="popover-actions">
        <button class="btn-tiny" @click="closeFieldEdit">cancel</button>
        <button class="btn-accent-tiny" @click="saveFieldEdit">save</button>
      </div>
    </div>

    <!-- New-run popover -->
    <div v-if="showNewRun" class="popover-backdrop" @click="showNewRun = false"></div>
    <div v-if="showNewRun" class="popover" :style="newRunPopoverStyle">
      <div class="popover-label">new run</div>
      <input
        v-model="newRunName"
        class="popover-input"
        placeholder="run name"
        @keydown.enter="confirmNewRun"
        @keydown.escape="showNewRun = false"
        autofocus
      />
      <div class="popover-actions">
        <button @click="showNewRun = false" class="btn-tiny">cancel</button>
        <button @click="confirmNewRun" class="btn-accent-tiny">create</button>
      </div>
    </div>

  </div>
</template>

<style scoped>
.pipeline {
  display: flex;
  flex-direction: column;
}

/* Run selector row */
.pipe-run-row {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 6px;
  border: 1px solid var(--border);
  border-radius: 2px 2px 0 0;
  background: var(--surface);
  font-size: 10px;
  border-bottom: none;
}
.pipe-run-select {
  flex: 1;
  min-width: 0;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--text);
  padding: 2px 4px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  outline: none;
}
.pipe-run-select:focus { border-color: var(--accent); }
.pipe-smoke-lbl {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 9px;
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
}
.pipe-smoke-lbl input { accent-color: var(--accent); margin: 0; }

/* Status row */
.pipe-status {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 8px;
  border: 1px solid var(--border);
  background: var(--bg);
  font-size: 10px;
  border-top: none;
  border-bottom: none;
}
.pipe-st-label {
  font-weight: 600;
  color: var(--text);
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.pipe-st-detail {
  color: var(--muted);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
}
.pipe-st-time {
  margin-left: auto;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: var(--text);
}
.pipe-st-eta {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: var(--accent);
}

.pipe-prog {
  height: 2px;
  background: var(--border);
  overflow: hidden;
}
.pipe-prog-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--green), var(--accent));
  transition: width 0.3s ease;
}

.pipe-err {
  font-size: 9px;
  padding: 3px 8px;
  border-radius: 0;
  border-left: none;
  border-right: none;
  border-top: none;
}

/* Step track */
.pipe-track {
  display: flex;
  align-items: flex-start;
  padding: 5px 8px 4px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-top: none;
  border-bottom: none;
}

.pipe-node {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  background: none;
  border: none;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 2px;
  flex-shrink: 0;
  transition: background 0.12s;
  min-width: 34px;
}
.pipe-node:hover { background: var(--surface-2); }

.pipe-node-ic {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  background: var(--bg);
  font-weight: 600;
  transition: all 0.12s;
  flex-shrink: 0;
}
.pipe-node-lb {
  font-size: 7px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  font-weight: 500;
  white-space: nowrap;
}

.pipe-node.ns-active .pipe-node-ic {
  border-color: var(--accent);
  color: var(--bg);
  background: var(--accent);
  font-weight: 700;
  box-shadow: 0 0 8px rgba(196,145,82,0.35);
}
.pipe-node.ns-active .pipe-node-lb { color: var(--accent); font-weight: 600; }

.pipe-node.ns-done .pipe-node-ic {
  border-color: var(--green);
  color: var(--bg);
  background: var(--green);
  font-weight: 700;
}
.pipe-node.ns-done .pipe-node-lb { color: var(--green); font-weight: 600; }

.pipe-node.ns-running .pipe-node-ic {
  border-color: var(--green);
  color: var(--bg);
  background: var(--green);
  font-weight: 700;
  animation: pulse 1.5s infinite;
  box-shadow: 0 0 8px rgba(93,158,93,0.35);
}

.pipe-node.ns-error .pipe-node-ic {
  border-color: var(--red);
  color: var(--red);
  background: rgba(185,82,76,0.08);
}
.pipe-node.ns-error .pipe-node-lb { color: var(--red); }

.pipe-node.ns-ready .pipe-node-ic {
  border-color: rgba(196,145,82,0.6);
  color: var(--accent);
  background: rgba(196,145,82,0.25);
  font-weight: 700;
}
.pipe-node.ns-ready .pipe-node-lb { color: var(--accent); font-weight: 600; }

.pipe-node.ns-skipped { opacity: 0.4; }
.pipe-node.ns-skipped .pipe-node-ic { border-style: dashed; }

.pipe-conn {
  flex: 1;
  height: 1px;
  background: var(--border);
  margin-top: 10px;
  margin-left: 2px;
  margin-right: 2px;
  transition: background 0.2s;
}
.pipe-conn.pipe-conn-done { background: rgba(93,158,93,0.45); }
.pipe-conn.pipe-conn-ready { background: rgba(196,145,82,0.3); }

/* Step panels */
.pipe-panel {
  border-radius: 0 0 2px 2px !important;
  border-top: none !important;
}

.pipe-ii {
  border: none;
  background: transparent;
  color: var(--text);
  outline: none;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
}
.pipe-ii:focus { border-bottom: 1px solid var(--accent) !important; }
.pipe-ii:hover { border-bottom: 1px solid var(--border) !important; }

.pipe-lbl {
  display: flex;
  align-items: center;
  gap: 2px;
  color: var(--muted);
  font-size: 9px;
  font-family: 'IBM Plex Mono', monospace;
}

.pipe-nav {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  gap: 4px;
  margin-top: 2px;
  padding-top: 4px;
  border-top: 1px solid var(--border);
}

.pipe-next {
  color: var(--accent) !important;
  border-color: rgba(196,145,82,0.3) !important;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
