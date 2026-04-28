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

// Algorithm from live configDetail
const currentAlgorithm = computed(() => {
  const agentGroup = training.configDetail.find(g => g.title === 'agent')
  if (!agentGroup) return 'linear_q'
  const field = agentGroup.fields.find(f => f[0] === 'algorithm')
  return field ? String(field[1]) : 'linear_q'
})
const isFrozenJepa = computed(() => currentAlgorithm.value === 'frozen_jepa_dqn')

async function onAlgorithmChange(e: Event) {
  const val = (e.target as HTMLSelectElement).value
  try {
    await api('/api/update-config', { overrides: [{ group: 'agent', key: 'algorithm', value: val }] })
    await training.refresh()
  } catch (err) {
    controlError.value = err instanceof Error ? err.message : String(err)
  }
}

function onNewRun(e: Event) {
  const detail = (e as CustomEvent).detail
  if (detail?.name) targetRunName.value = detail.name
}
onMounted(() => window.addEventListener('new-run', onNewRun))
onUnmounted(() => window.removeEventListener('new-run', onNewRun))

// Live clock for elapsed time
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
  if (s.algorithm) parts.push(String(s.algorithm))
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
  if (!checkpoints.length) {
    selectedCheckpoint.value = ''
    return
  }
  const files = checkpoints.map(c => c.file)
  if (!selectedCheckpoint.value || !files.includes(selectedCheckpoint.value)) {
    selectedCheckpoint.value = files[files.length - 1]
  }
}, { immediate: true })

onMounted(async () => {
  try {
    const defaults = await getJson<{
      default_steps: number
      learning_starts: number
      batch_size: number | null
      dashboard_every: number
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
  if (!experiment) {
    controlError.value = 'select or create a run first'
    return
  }
  const ckptStep = checkpointStep()
  const totalSteps = ckptStep + Number(additionalSteps.value)
  const payload: Record<string, unknown> = {
    experiment,
    steps: totalSteps,
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
    if (training.isEvaluating) {
      await api('/api/eval/stop')
    } else {
      await api('/api/train/stop')
    }
  } catch (e) {
    controlError.value = e instanceof Error ? e.message : String(e)
    console.error(e)
  }
  training.refresh()
}

async function watchAiPlay() {
  controlError.value = ''
  if (!selectedCheckpoint.value) {
    controlError.value = 'select a checkpoint first'
    return
  }
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

// --- Collect Baseline ---
const collectOpen = ref(false)
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
  } catch (e) {
    collectError.value = e instanceof Error ? e.message : String(e)
  }
}

async function stopCollect() {
  try {
    await training.stopCollect()
  } catch (e) {
    collectError.value = e instanceof Error ? e.message : String(e)
  }
}

// --- Train World Model ---
const worldOpen = ref(false)
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
  } catch (e) {
    worldError.value = e instanceof Error ? e.message : String(e)
  }
}

async function stopWorldTraining() {
  try {
    await training.stopWorldTraining()
  } catch (e) {
    worldError.value = e instanceof Error ? e.message : String(e)
  }
}

// --- Tools ---
const toolsOpen = ref(false)
const smokeSteps = ref(2000)
const smokeLr = ref(0.03)
const smokeSeed = ref(0)
const smokeResult = ref<{ passed?: boolean; improvement?: number; initial_loss?: number; final_loss?: number; error?: string } | null>(null)
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
  } finally {
    smokeRunning.value = false
  }
}
</script>

<template>
  <div class="train-controls" id="controlGroup" :data-run-dir="runs.runDir">
    <!-- Status row -->
    <div class="tc-status-row">
      <span :class="dotClass"></span>
      <span class="tc-status-text">{{ statusText }}</span>
      <span class="tc-status-detail">{{ statusDetail }}</span>
      <template v-if="isTraining && elapsed != null">
        <span class="tc-elapsed" title="Elapsed">{{ fmtDuration(elapsed) }}</span>
        <span v-if="eta != null" class="tc-eta" title="Estimated remaining">~{{ fmtDuration(eta) }}</span>
      </template>
    </div>
    <div v-if="isTraining && progress != null" class="tc-progress-track" :title="(progress * 100).toFixed(1) + '% complete'">
      <div class="tc-progress-fill" :style="{ width: (progress * 100) + '%' }"></div>
    </div>

    <!-- Algorithm + headed row -->
    <div class="tc-row tc-algo-row">
      <div class="field" title="RL algorithm">
        algo
        <select :value="currentAlgorithm" @change="onAlgorithmChange" :disabled="busy" title="Algorithm">
          <option value="dqn">dqn</option>
          <option value="frozen_jepa_dqn">frozen jepa dqn</option>
          <option value="linear_q">linear q</option>
        </select>
      </div>
      <label class="tc-check" title="Show browser window during training">
        <input type="checkbox" v-model="headed" :disabled="busy" />
        headed
      </label>
    </div>

    <!-- JEPA checkpoint (frozen_jepa_dqn only) -->
    <div class="tc-row" v-if="isFrozenJepa">
      <div class="field tc-field-wide" title="Path to pretrained JEPA encoder checkpoint (.pt)">
        jepa ckpt
        <input v-model="jepaCheckpoint" type="text" placeholder="/path/to/latest.pt" title="JEPA checkpoint path" class="tc-input-wide" />
      </div>
    </div>

    <!-- Training params -->
    <div class="tc-row">
      <div class="field" title="Additional environment steps to train">+steps <input v-model.number="additionalSteps" type="number" min="1" title="Additional steps"></div>
      <div class="field" title="Steps before model updates begin">warmup <input v-model.number="warmup" type="number" min="0" title="Warmup steps"></div>
      <div v-if="hasCheckpoint && checkpointStep() > 0" class="tc-hint">→ {{ checkpointStep() + additionalSteps }} total</div>
    </div>
    <div class="tc-row">
      <div class="field" title="Minibatch size">batch <input v-model="batch" type="number" min="1" placeholder="auto" title="Batch size override"></div>
      <div class="field" title="Rewrite dashboard every N steps">log&nbsp;every <input v-model.number="logEvery" type="number" min="1" title="Dashboard write interval"></div>
      <div class="field" title="Learning rate override (leave blank to use config)">lr <input v-model="lr" type="number" step="any" placeholder="auto" title="Learning rate override" style="width:56px"></div>
    </div>
    <div class="tc-row cg-ckpt" v-if="hasCheckpoint">
      <span class="cg-ckpt-label">checkpoint</span>
      <select v-model="selectedCheckpoint" title="Available model checkpoints">
        <option v-for="c in availableCheckpoints" :key="c.file" :value="c.file">{{ c.label }}</option>
      </select>
    </div>
    <div class="tc-row tc-error" v-if="controlError">{{ controlError }}</div>
    <div class="tc-row tc-actions">
      <button @click="startTraining" v-if="!busy" class="btn-accent" title="Start training">train</button>
      <button @click="watchAiPlay" v-if="!busy && hasCheckpoint" class="btn-accent" title="Watch AI play">watch AI play</button>
      <button @click="stopTraining" v-if="busy" class="btn-danger" title="Stop training/eval">stop</button>
    </div>

    <!-- Collect Baseline section -->
    <div class="tc-section">
      <button class="tc-section-toggle" @click="collectOpen = !collectOpen">
        <span class="tc-section-arrow" :class="{ open: collectOpen }">▶</span>
        collect baseline
        <span v-if="training.isCollecting" class="tc-section-badge running">
          {{ training.collectJob?.episodes_done }}/{{ training.collectJob?.episodes_target }}
        </span>
        <span v-else-if="training.collectJob?.status === 'completed'" class="tc-section-badge done">done</span>
        <span v-else-if="training.collectJob?.status === 'error'" class="tc-section-badge err">error</span>
      </button>
      <div v-if="collectOpen" class="tc-section-body">
        <div class="tc-row">
          <div class="field" title="Number of random episodes">episodes <input v-model.number="collectEpisodes" type="number" min="1" /></div>
          <div class="field" title="Max steps per episode">max steps <input v-model.number="collectMaxSteps" type="number" min="1" /></div>
        </div>
        <div class="tc-row">
          <div class="field tc-field-wide" title="Experiment name (optional)">
            name <input v-model="collectExperiment" type="text" placeholder="auto" class="tc-input-wide" />
          </div>
        </div>
        <div v-if="training.isCollecting" class="tc-section-status">
          collecting… ep {{ training.collectJob?.episodes_done }}/{{ training.collectJob?.episodes_target }}
          <span v-if="training.collectJob?.mean_score"> · mean {{ fmtNum(training.collectJob.mean_score) }}</span>
        </div>
        <div v-if="training.collectJob?.status === 'error'" class="tc-row tc-error">{{ training.collectJob.error }}</div>
        <div class="tc-row tc-actions">
          <button v-if="!training.isCollecting" @click="startCollect" class="btn-accent" :disabled="busy">collect</button>
          <button v-if="training.isCollecting" @click="stopCollect" class="btn-danger">stop</button>
        </div>
        <div class="tc-row tc-error" v-if="collectError">{{ collectError }}</div>
      </div>
    </div>

    <!-- Train World Model section -->
    <div class="tc-section">
      <button class="tc-section-toggle" @click="worldOpen = !worldOpen">
        <span class="tc-section-arrow" :class="{ open: worldOpen }">▶</span>
        train world model
        <span v-if="training.isWorldTraining" class="tc-section-badge running">running</span>
        <span v-else-if="training.worldJob?.status === 'completed'" class="tc-section-badge done">done</span>
        <span v-else-if="training.worldJob?.status === 'error'" class="tc-section-badge err">error</span>
      </button>
      <div v-if="worldOpen" class="tc-section-body">
        <div class="tc-row">
          <div class="field" title="Gradient update steps">steps <input v-model.number="worldSteps" type="number" min="1" /></div>
          <div class="field" title="Browser steps for random data before training (optional)">pre-collect <input v-model="worldCollectSteps" type="number" min="0" placeholder="auto" style="width:56px" /></div>
        </div>
        <div class="tc-row">
          <div class="field" title="Batch size">batch <input v-model="worldBatch" type="number" min="1" placeholder="auto" /></div>
          <div class="field" title="Learning rate">lr <input v-model="worldLr" type="number" step="any" placeholder="auto" style="width:56px" /></div>
          <div class="field" title="Dashboard write interval">log <input v-model.number="worldDashEvery" type="number" min="1" style="width:44px" /></div>
        </div>
        <div v-if="training.isWorldTraining" class="tc-section-status">
          training world model…
          <span v-if="training.worldJob"> · {{ training.worldJob.run_name }}</span>
        </div>
        <div v-if="training.worldJob?.status === 'error'" class="tc-row tc-error">{{ training.worldJob.error }}</div>
        <div class="tc-row tc-actions">
          <button v-if="!training.isWorldTraining" @click="startWorldTraining" class="btn-accent">train world</button>
          <button v-if="training.isWorldTraining" @click="stopWorldTraining" class="btn-danger">stop</button>
        </div>
        <div class="tc-row tc-error" v-if="worldError">{{ worldError }}</div>
      </div>
    </div>

    <!-- Tools section -->
    <div class="tc-section">
      <button class="tc-section-toggle" @click="toolsOpen = !toolsOpen">
        <span class="tc-section-arrow" :class="{ open: toolsOpen }">▶</span>
        tools
      </button>
      <div v-if="toolsOpen" class="tc-section-body">
        <!-- Validate config -->
        <div class="tc-tool-row">
          <button @click="runValidate" class="btn-tiny">validate config</button>
          <span v-if="validateResult" :class="validateResult.ok ? 'tc-badge-ok' : 'tc-badge-err'">
            {{ validateResult.ok ? `✓ ${validateResult.game} / ${validateResult.algorithm}` : `✗ ${validateResult.error}` }}
          </span>
        </div>
        <!-- ML smoke test -->
        <div class="tc-tool-row">
          <div class="field" title="SGD updates">steps <input v-model.number="smokeSteps" type="number" min="1" style="width:56px" /></div>
          <div class="field" title="Learning rate">lr <input v-model.number="smokeLr" type="number" step="any" style="width:48px" /></div>
          <div class="field" title="Random seed">seed <input v-model.number="smokeSeed" type="number" min="0" style="width:40px" /></div>
          <button @click="runSmoke" class="btn-tiny" :disabled="smokeRunning">{{ smokeRunning ? '…' : 'ml smoke' }}</button>
        </div>
        <div v-if="smokeResult" :class="smokeResult.passed ? 'tc-badge-ok' : 'tc-badge-err'" class="tc-tool-result">
          <template v-if="!smokeResult.error">
            {{ smokeResult.passed ? '✓ passed' : '✗ failed' }}
            · Δloss {{ fmtNum(smokeResult.improvement) }}
            ({{ fmtNum(smokeResult.initial_loss) }} → {{ fmtNum(smokeResult.final_loss) }})
          </template>
          <template v-else>✗ {{ smokeResult.error }}</template>
        </div>
      </div>
    </div>
  </div>
</template>
