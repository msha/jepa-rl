<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import { useTrainingStore } from "../stores/training";
import { useRunsStore } from "../stores/runs";
import { useConfigStore } from "../stores/config";
import { api, getJson } from "../api/client";
import VDropdown from "./VDropdown.vue";
import type { Job, EvalJob } from "../stores/training";

const props = defineProps<{
    job: Job | null;
    evalJob: EvalJob | null;
    summary: Record<string, unknown>;
    latestStep: Record<string, unknown>;
}>();

const training = useTrainingStore();
const runs = useRunsStore();
const configStore = useConfigStore();

const targetRunName = ref("");
const additionalSteps = ref(500);
const warmup = ref(50);
const batch = ref("");
const logEvery = ref(5);
const lr = ref("");
const headed = ref(false);
const jepaCheckpoint = ref("");
const selectedCheckpoint = ref("");
const controlError = ref("");

const pipeRunOptions = computed(() => [
    { value: "", label: "select run or create a new by starting from here..." },
    ...runs.runs.map((r) => ({ value: r.name, label: fmtRun(r) })),
]);

const checkpointOptions = computed(() =>
    availableCheckpoints.value.map((c) => ({ value: c.file, label: c.label })),
);

const currentAlgorithm = computed(() => {
    const mi = training.modelInfo;
    return String(mi.algorithm || "linear_q");
});
const isFrozenJepa = computed(
    () => currentAlgorithm.value === "frozen_jepa_dqn",
);
watch(isFrozenJepa, (frozen) => {
    if (!frozen && activeStep.value === 2) activeStep.value = 1;
});

function onNewRun(e: Event) {
    const detail = (e as CustomEvent).detail;
    if (detail?.name) targetRunName.value = detail.name;
}
onMounted(() => window.addEventListener("new-run", onNewRun));
onUnmounted(() => window.removeEventListener("new-run", onNewRun));

const now = ref(Math.floor(Date.now() / 1000));
let clock: ReturnType<typeof setInterval> | null = null;
onMounted(() => {
    clock = setInterval(() => {
        now.value = Math.floor(Date.now() / 1000);
    }, 1000);
});
onUnmounted(() => {
    if (clock) clearInterval(clock);
});

const busy = computed(() => training.isTraining || training.isEvaluating);
const isTraining = computed(
    () =>
        !!props.job?.running ||
        props.job?.status === "running" ||
        props.job?.status === "starting",
);
const isEvaluating = computed(
    () =>
        !isTraining.value &&
        (!!props.evalJob?.running || props.evalJob?.status === "running"),
);

const dotClass = computed(() => {
    if (isTraining.value) return "tc-dot running";
    if (isEvaluating.value) return "tc-dot evaluating";
    if (props.job?.status === "error" || props.evalJob?.status === "error")
        return "tc-dot error";
    if (props.job?.status === "completed") return "tc-dot stopped";
    return "tc-dot idle";
});

const statusText = computed(() => {
    if (isTraining.value) return "training";
    if (isEvaluating.value) return "evaluating";
    if (props.job?.status === "completed") return "completed";
    if (props.job?.status === "stopped") return "stopped";
    if (props.job?.status === "error") return "error";
    if (props.evalJob?.status === "error") return "eval error";
    return "idle";
});

const statusDetail = computed(() => {
    const s = props.summary || {};
    const parts: string[] = [];
    if (s.algorithm) {
        const display: Record<string, string> = {
            pixel_dqn: "dqn",
            PixelDQN: "dqn",
            frozen_jepa_dqn: "frozen jepa dqn",
            FrozenJEPADQN: "frozen jepa dqn",
            linear_q: "linear q",
            LinearQ: "linear q",
        };
        parts.push(display[String(s.algorithm)] || String(s.algorithm));
    }
    if (s.episodes) parts.push(`${s.episodes} eps`);
    if (s.best_score != null) parts.push("best:" + fmtNum(s.best_score));
    if (s.steps) parts.push(fmtNum(s.steps) + " steps");
    return parts.join(" · ");
});

const startTs = computed(() => {
    if (!props.job?.started_at) return null;
    return props.job.started_at > 1e12
        ? props.job.started_at / 1000
        : props.job.started_at;
});

const elapsed = computed(() => {
    if (!isTraining.value || startTs.value == null) return null;
    return Math.max(0, now.value - Math.floor(startTs.value));
});

const currentStep = computed(() =>
    Number(props.latestStep?.step ?? props.summary?.steps ?? 0),
);

const eta = computed(() => {
    if (
        !isTraining.value ||
        startTs.value == null ||
        !props.job?.requested_steps
    )
        return null;
    if (currentStep.value <= 0) return null;
    const elapsedSec = Math.max(1, now.value - Math.floor(startTs.value));
    const rate = currentStep.value / elapsedSec;
    const remaining = props.job.requested_steps - currentStep.value;
    if (remaining <= 0) return null;
    return remaining / rate;
});

const progress = computed(() => {
    if (!isTraining.value || !props.job?.requested_steps) return null;
    return Math.min(1, currentStep.value / props.job.requested_steps);
});

function fmtDuration(sec: number | null): string {
    if (sec == null || sec < 0) return "";
    sec = Math.floor(sec);
    if (sec < 60) return `${sec}s`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m < 60) return `${m}m ${s}s`;
    const hr = Math.floor(m / 60);
    return `${hr}h ${m % 60}m`;
}

function fmtNum(v: unknown): string {
    if (v == null || typeof v !== "number") return "—";
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(2);
    return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

function fmtRun(r: {
    name: string;
    experiment_name?: string;
    algorithm?: string;
    best_score?: number;
    steps?: number;
}): string {
    const label = r.experiment_name || r.name;
    const parts = [label];
    if (r.algorithm) parts.push(r.algorithm);
    if (r.best_score != null) parts.push("best:" + fmtNum(r.best_score));
    if (r.steps != null) parts.push(r.steps + " steps");
    return parts.join(" · ");
}

function onRunChange() {
    runs.loadRunDetail(runs.selectedRun);
}

async function deleteSelectedRun() {
    if (!runs.selectedRun) return;
    if (!confirm(`Delete run "${runs.selectedRun}" and all its data?`)) return;
    try {
        await runs.deleteRun(runs.selectedRun);
    } catch (e) {
        controlError.value = e instanceof Error ? e.message : String(e);
    }
}

// New-run popover
const showNewRun = ref(false);
const newRunName = ref("");
const newRunPopoverStyle = ref<Record<string, string>>({});

function openNewRun(e: MouseEvent) {
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    newRunName.value = `run_${ts}`;
    newRunPopoverStyle.value = anchorBelow(e.currentTarget as HTMLElement, 200);
    showNewRun.value = true;
}

function confirmNewRun() {
    const name = newRunName.value.trim();
    if (!name) return;
    runs.selectedRun = "";
    runs.checkpoints = [];
    runs.runDir = "";
    runs.runConfigDetail = null;
    window.dispatchEvent(new CustomEvent("new-run", { detail: { name } }));
    targetRunName.value = name;
    showNewRun.value = false;
}

const availableCheckpoints = computed(() => {
    if (runs.checkpoints.length) return runs.checkpoints;
    return training.runDir?.checkpoints ?? [];
});
const evalRunDir = computed(() => runs.runDir || training.runDir?.dir || "");
const hasCheckpoint = computed(() => availableCheckpoints.value.length > 0);

function checkpointStep(): number {
    if (!selectedCheckpoint.value) return 0;
    const m = selectedCheckpoint.value.match(/(\d+)/);
    return m ? parseInt(m[1], 10) : 0;
}

watch(
    availableCheckpoints,
    (checkpoints) => {
        if (!checkpoints.length) {
            selectedCheckpoint.value = "";
            return;
        }
        const files = checkpoints.map((c) => c.file);
        if (
            !selectedCheckpoint.value ||
            !files.includes(selectedCheckpoint.value)
        ) {
            selectedCheckpoint.value = files[files.length - 1];
        }
    },
    { immediate: true },
);

onMounted(async () => {
    try {
        const defaults = await getJson<{
            default_steps: number;
            learning_starts: number;
            batch_size: number | null;
            dashboard_every: number;
        }>("/api/defaults");
        additionalSteps.value = defaults.default_steps || 500;
        warmup.value = defaults.learning_starts ?? 50;
        batch.value =
            defaults.batch_size != null ? String(defaults.batch_size) : "";
        logEvery.value = defaults.dashboard_every || 5;
    } catch {
        /* use defaults */
    }
});

async function startTraining() {
    controlError.value = "";
    const experiment = targetRunName.value || runs.selectedRun || "";
    if (!experiment) {
        controlError.value = "select or create a run first";
        return;
    }
    const ckptStep = checkpointStep();
    const totalSteps = ckptStep + Number(additionalSteps.value);
    const payload: Record<string, unknown> = {
        experiment,
        steps: totalSteps,
        learning_starts: Number(warmup.value),
        batch_size: batch.value || null,
        dashboard_every: Number(logEvery.value) || 5,
        headed: headed.value,
    };
    if (lr.value) payload.lr = Number(lr.value);
    if (isFrozenJepa.value && jepaCheckpoint.value)
        payload.jepa_checkpoint = jepaCheckpoint.value;
    try {
        await api("/api/train/start", payload);
    } catch (e) {
        controlError.value = e instanceof Error ? e.message : String(e);
        console.error(e);
    }
    training.refresh();
    setTimeout(() => runs.loadRuns(), 500);
}

async function stopTraining() {
    controlError.value = "";
    try {
        if (training.isEvaluating) await api("/api/eval/stop");
        else await api("/api/train/stop");
    } catch (e) {
        controlError.value = e instanceof Error ? e.message : String(e);
        console.error(e);
    }
    training.refresh();
}

async function watchAiPlay() {
    controlError.value = "";
    if (!selectedCheckpoint.value) {
        controlError.value = "select a checkpoint first";
        return;
    }
    const payload: Record<string, unknown> = {
        episodes: 3,
        checkpoint: selectedCheckpoint.value,
    };
    if (evalRunDir.value) payload.run_dir = evalRunDir.value;
    if (isFrozenJepa.value && jepaCheckpoint.value)
        payload.jepa_checkpoint = jepaCheckpoint.value;
    try {
        await api("/api/eval/start", payload);
    } catch (e) {
        controlError.value = e instanceof Error ? e.message : String(e);
        console.error(e);
    }
    training.refresh();
}

const collectEpisodes = ref(5);
const collectMaxSteps = ref(200);
const collectError = ref("");

// Auto-generate collection name: <game>-rand-act-<next_free_int>
const autoCollectName = computed(() => {
    const game = training.gameName || "game";
    const existingNames = new Set(runs.collectedDatasets.map((d) => d.name));
    let n = 1;
    while (existingNames.has(`${game}-rand-act-${n}`)) n++;
    return `${game}-rand-act-${n}`;
});

const datasetOptions = computed(() => {
    const opts = [{ value: "", label: "none" }];
    for (const d of runs.collectedDatasets) {
        const parts = [d.name];
        if (d.episodes) parts.push(`${d.episodes} eps`);
        if (d.mean_score != null) parts.push(`avg ${fmtNum(d.mean_score)}`);
        opts.push({ value: d.name, label: parts.join(" · ") });
    }
    return opts;
});
const selectedDataset = ref("");

async function startCollect() {
    collectError.value = "";
    const experiment = autoCollectName.value;
    try {
        await training.startCollect({
            experiment,
            episodes: collectEpisodes.value,
            max_steps: collectMaxSteps.value,
            headed: headed.value,
        });
        runs.loadCollectedDatasets();
    } catch (e) {
        collectError.value = e instanceof Error ? e.message : String(e);
    }
}

async function stopCollect() {
    try {
        await training.stopCollect();
    } catch (e) {
        collectError.value = e instanceof Error ? e.message : String(e);
    }
}

const worldSteps = ref(1000);
const worldCollectSteps = ref("");
const worldBatch = ref("");
const worldLr = ref("");
const worldDashEvery = ref(25);
const worldError = ref("");

async function startWorldTraining() {
    worldError.value = "";
    try {
        await training.startWorldTraining({
            experiment:
                (targetRunName.value || runs.selectedRun || undefined) &&
                `${targetRunName.value || runs.selectedRun}_world`,
            steps: worldSteps.value,
            collect_steps: worldCollectSteps.value
                ? Number(worldCollectSteps.value)
                : undefined,
            batch_size: worldBatch.value ? Number(worldBatch.value) : undefined,
            lr: worldLr.value ? Number(worldLr.value) : undefined,
            dashboard_every: worldDashEvery.value,
            headed: headed.value,
        });
    } catch (e) {
        worldError.value = e instanceof Error ? e.message : String(e);
    }
}

async function stopWorldTraining() {
    try {
        await training.stopWorldTraining();
    } catch (e) {
        worldError.value = e instanceof Error ? e.message : String(e);
    }
}

// Field edit popover
const editingField = ref<string | null>(null);
const editValue = ref("");
const editPopoverStyle = ref<Record<string, string>>({});

function displayVal(key: string): string {
    switch (key) {
        case "collectEpisodes":
            return String(collectEpisodes.value);
        case "collectMaxSteps":
            return String(collectMaxSteps.value);
        case "worldSteps":
            return String(worldSteps.value);
        case "worldBatch":
            return worldBatch.value;
        case "worldLr":
            return worldLr.value;
        case "worldCollectSteps":
            return worldCollectSteps.value;
        case "worldDashEvery":
            return String(worldDashEvery.value);
        case "warmup":
            return String(warmup.value);
        case "additionalSteps":
            return String(additionalSteps.value);
        case "batch":
            return batch.value;
        case "lr":
            return lr.value;
        case "logEvery":
            return String(logEvery.value);
        case "jepaCheckpoint":
            return jepaCheckpoint.value;
        default:
            return "";
    }
}

function openFieldEdit(key: string, event: MouseEvent) {
    editingField.value = key;
    editValue.value = displayVal(key);
    editPopoverStyle.value = anchorBelow(
        event.currentTarget as HTMLElement,
        200,
    );
}

function saveFieldEdit() {
    if (!editingField.value) return;
    const key = editingField.value;
    const val = editValue.value;
    const n = Number(val);
    switch (key) {
        case "collectEpisodes":
            collectEpisodes.value = n;
            break;
        case "collectMaxSteps":
            collectMaxSteps.value = n;
            break;
        case "worldSteps":
            worldSteps.value = n;
            break;
        case "worldBatch":
            worldBatch.value = val;
            break;
        case "worldLr":
            worldLr.value = val;
            break;
        case "worldCollectSteps":
            worldCollectSteps.value = val;
            break;
        case "worldDashEvery":
            worldDashEvery.value = n;
            break;
        case "warmup":
            warmup.value = n;
            break;
        case "additionalSteps":
            additionalSteps.value = n;
            break;
        case "batch":
            batch.value = val;
            break;
        case "lr":
            lr.value = val;
            break;
        case "logEvery":
            logEvery.value = n;
            break;
        case "jepaCheckpoint":
            jepaCheckpoint.value = val;
            break;
    }
    editingField.value = null;
}

function closeFieldEdit() {
    editingField.value = null;
}

// Model config edit via config store overrides
async function saveConfigEdit(group: string, key: string, value: string) {
    try {
        await configStore.applyOverrides([{ group, key, value }]);
        await training.refresh();
    } catch (e) {
        console.error("Config update failed:", e);
    }
}

// Pipeline
const activeStep = ref(0);

type StepStatus =
    | "pending"
    | "ready"
    | "done"
    | "running"
    | "error"
    | "skipped";

const stepStatus = computed((): StepStatus[] => {
    // Step 0: Model — always "ready" (or "done" once validated)
    const s0: StepStatus = "ready";
    const s1: StepStatus = training.isCollecting
        ? "running"
        : training.collectJob?.status === "completed"
          ? "done"
          : training.collectJob?.status === "error"
            ? "error"
            : "ready";
    const s2: StepStatus = !isFrozenJepa.value
        ? "skipped"
        : training.isWorldTraining
          ? "running"
          : training.worldJob?.status === "completed"
            ? "done"
            : training.worldJob?.status === "error"
              ? "error"
              : "ready";
    const s3: StepStatus = isTraining.value
        ? "running"
        : props.job?.status === "completed"
          ? "done"
          : props.job?.status === "error"
            ? "error"
            : targetRunName.value || runs.selectedRun
              ? "ready"
              : "pending";
    return [s0, s1, s2, s3];
});

const STEP_LABELS = ["model", "data", "jepa", "train"];
const STEP_TITLES = [
    "Model Setup",
    "Collect Data",
    "Pretrain JEPA",
    "Train RL Agent",
];

const visibleSteps = computed(() =>
    isFrozenJepa.value ? [0, 1, 2, 3] : [0, 1, 3],
);

function nextStep(current: number): number {
    const idx = visibleSteps.value.indexOf(current);
    return idx >= 0 && idx < visibleSteps.value.length - 1
        ? visibleSteps.value[idx + 1]
        : current;
}
function prevStep(current: number): number {
    const idx = visibleSteps.value.indexOf(current);
    return idx > 0 ? visibleSteps.value[idx - 1] : current;
}

function nodeIcon(status: StepStatus, idx: number): string {
    if (status === "done") return "✓";
    if (status === "error") return "✗";
    if (status === "skipped") return "–";
    if (status === "running") return "●";
    return String(idx + 1);
}

// Passive config validation
const validationResult = ref<{
    ok: boolean;
    game?: string;
    algorithm?: string;
    error?: string;
} | null>(null);
watch(
    () => configStore.currentPath,
    async () => {
        validationResult.value = null;
        if (configStore.currentPath) {
            validationResult.value = await training.validateConfig(
                configStore.currentPath,
            );
        }
    },
    { immediate: true },
);

// Load collected datasets on mount
onMounted(() => {
    runs.loadCollectedDatasets();
});

// Shared popover positioning helper
function anchorBelow(el: HTMLElement, minWidth = 180): Record<string, string> {
    const rect = el.getBoundingClientRect();
    const w = Math.max(minWidth, rect.width);
    let top = rect.bottom + 4;
    let left = rect.left;
    if (top + 110 > window.innerHeight - 8) top = rect.top - 110 - 4;
    if (left + w > window.innerWidth - 8) left = rect.right - w;
    if (left < 8) left = 8;
    return { top: `${top}px`, left: `${left}px`, width: `${w}px` };
}

// Model info helpers
const mi = computed(() => training.modelInfo);
const obsLabel = computed(() => {
    const w = mi.value.observation_width ?? "?";
    const h = mi.value.observation_height ?? "?";
    return `${w}×${h}`;
});
const colorLabel = computed(() => {
    const gray = mi.value.grayscale;
    const fs = mi.value.frame_stack ?? "?";
    return gray ? `grayscale · ${fs} frames` : `color · ${fs} frames`;
});
const encoderLabel = computed(() => {
    const ch = mi.value.encoder_channels as number[] | undefined;
    const t = mi.value.encoder_type ?? "?";
    if (ch?.length) return `${t} [${ch.join("→")}]`;
    return String(t);
});
const predictorLabel = computed(() => {
    const t = mi.value.predictor_type ?? "?";
    const d = mi.value.predictor_depth;
    const h = mi.value.predictor_heads;
    const hid = mi.value.predictor_hidden;
    const parts = [String(t)];
    if (d) parts.push(`depth ${d}`);
    if (h) parts.push(`${h} heads`);
    if (hid) parts.push(`dim ${hid}`);
    return parts.join(" · ");
});
const qNetLabel = computed(() => {
    const dims = mi.value.q_hidden_dims as number[] | undefined;
    const dueling = mi.value.q_dueling;
    const parts: string[] = [];
    if (dims?.length) parts.push(dims.join("×"));
    if (dueling) parts.push("dueling");
    return parts.join(" · ") || "—";
});
const actionLabel = computed(() => {
    const n = mi.value.num_actions ?? 0;
    const keys = mi.value.action_keys as string[] | undefined;
    if (keys?.length) return `${n} actions`;
    return String(n);
});

// Algorithm display names
const algoDisplay: Record<string, string> = {
    dqn: "Pixel DQN",
    frozen_jepa_dqn: "Frozen JEPA + DQN",
    linear_q: "Linear Q",
    joint_jepa_dqn: "Joint JEPA + DQN",
};

// Dynamic algorithm descriptions
const algoDescriptions: Record<string, { short: string; tip: string }> = {
    dqn: {
        short: "End-to-end pixel Q-learning with dueling architecture, Double DQN, and epsilon-greedy exploration",
        tip: "Pixel DQN learns directly from raw pixel frames. Uses a convolutional encoder to extract features, a dueling Q-network to estimate action values (separating state-value and advantage), Double DQN to reduce overestimation, and epsilon-greedy exploration that decays from random to greedy over time. Simple and reliable baseline.",
    },
    frozen_jepa_dqn: {
        short: "Pretrained JEPA encoder (frozen) + Q-head trained on latent representations for sample-efficient learning",
        tip: "Two-stage approach: first pretrain a JEPA world model to learn latent representations of the environment dynamics (Step 3). Then freeze the encoder and train only a Q-value head on top of the learned latents. The frozen encoder provides rich features that make Q-learning much more sample-efficient than learning from pixels directly.",
    },
    linear_q: {
        short: "Minimal linear Q-learner for smoke testing and pipeline validation",
        tip: "A trivial linear Q-function approximator with no neural network. Used only for smoke testing to verify the training pipeline works end-to-end without the cost of a real model. Not intended for actual gameplay.",
    },
    joint_jepa_dqn: {
        short: "Jointly trains JEPA encoder and DQN head simultaneously with shared gradients",
        tip: "Trains the JEPA world model encoder and the DQN Q-head jointly in a single training loop. The encoder learns representations that are simultaneously good for predicting future states and for estimating action values. More complex than frozen JEPA but potentially more powerful.",
    },
};
const algoShortDesc = computed(
    () =>
        algoDescriptions[currentAlgorithm.value]?.short ??
        "RL learning algorithm",
);
const algoTip = computed(() => {
    const d = algoDescriptions[currentAlgorithm.value];
    if (d) return d.tip;
    const all = Object.entries(algoDescriptions)
        .map(([k, v]) => `${algoDisplay[k] || k}: ${v.short}`)
        .join("\n\n");
    return all;
});

// Editable model field popover
const editingConfigField = ref<string | null>(null);
const configEditValue = ref("");
const configEditMeta = ref<{ group: string; key: string }>({
    group: "",
    key: "",
});
const configPopoverStyle = ref<Record<string, string>>({});

function openConfigEdit(
    group: string,
    key: string,
    currentVal: unknown,
    event: MouseEvent,
) {
    editingConfigField.value = key;
    configEditMeta.value = { group, key };
    configEditValue.value = String(currentVal ?? "");
    configPopoverStyle.value = anchorBelow(
        event.currentTarget as HTMLElement,
        200,
    );
}

function saveConfigFieldEdit() {
    if (!editingConfigField.value) return;
    saveConfigEdit(
        configEditMeta.value.group,
        configEditMeta.value.key,
        configEditValue.value,
    );
    editingConfigField.value = null;
}

function closeConfigFieldEdit() {
    editingConfigField.value = null;
}
</script>

<template>
    <div class="pipeline" id="controlGroup" :data-run-dir="runs.runDir">
        <!-- Run selector row -->
        <div class="pipe-run-row">
            <label class="pipe-smoke-lbl" title="Include smoke test runs">
                <input
                    type="checkbox"
                    v-model="runs.showSmoke"
                    @change="runs.loadRuns()"
                />
                smoke
            </label>
            <VDropdown
                v-model="runs.selectedRun"
                :options="pipeRunOptions"
                @change="onRunChange"
                title="Select a training run"
                full-width
            />
            <button
                v-if="!runs.selectedRun"
                @click="openNewRun"
                class="btn-tiny"
            >
                + new
            </button>
            <button
                v-if="runs.selectedRun"
                @click="deleteSelectedRun"
                class="btn-danger-tiny"
            >
                del
            </button>
        </div>

        <!-- Status row -->
        <div class="pipe-status">
            <span :class="dotClass"></span>
            <span class="pipe-st-label">{{ statusText }}</span>
            <span class="pipe-st-detail">{{ statusDetail }}</span>
            <template v-if="isTraining && elapsed != null">
                <span class="pipe-st-time">{{ fmtDuration(elapsed) }}</span>
                <span v-if="eta != null" class="pipe-st-eta"
                    >~{{ fmtDuration(eta) }}</span
                >
            </template>
            <button v-if="busy" @click="stopTraining" class="btn-danger-tiny">
                stop
            </button>
        </div>
        <div v-if="isTraining && progress != null" class="pipe-prog">
            <div
                class="pipe-prog-fill"
                :style="{ width: progress * 100 + '%' }"
            ></div>
        </div>
        <div class="tc-error pipe-err" v-if="controlError">
            {{ controlError }}
        </div>

        <!-- Step track -->
        <div class="pipe-track">
            <template v-for="(si, vi) in visibleSteps" :key="si">
                <button
                    class="pipe-node"
                    :class="[
                        'ns-' + stepStatus[si],
                        { 'ns-active': activeStep === si },
                    ]"
                    @click="activeStep = si"
                    :title="STEP_TITLES[si]"
                >
                    <span class="pipe-node-ic">{{
                        nodeIcon(stepStatus[si], si)
                    }}</span>
                    <span class="pipe-node-lb">{{ STEP_LABELS[si] }}</span>
                </button>
                <div
                    v-if="vi < visibleSteps.length - 1"
                    class="pipe-conn"
                    :class="{
                        'pipe-conn-done': stepStatus[si] === 'done',
                        'pipe-conn-ready':
                            stepStatus[si] === 'ready' ||
                            stepStatus[si] === 'running',
                    }"
                ></div>
            </template>
        </div>

        <!-- Step 0: Model Setup -->
        <div v-show="activeStep === 0" class="pipe-panel pipe-panel-model">
            <div class="pipe-panel-desc">
                Review and configure the model architecture and observation
                settings before collecting data or training.
            </div>

            <!-- Config validation error only -->
            <div
                v-if="validationResult && !validationResult.ok"
                class="pipe-validation"
            >
                <span class="pipe-val-err">✗ {{ validationResult.error }}</span>
            </div>

            <!-- Model info as settings table -->
            <div class="settings-table mi-table">
                <span class="sk">algorithm</span>
                <div class="sv mi-sv">
                    <span
                        class="mi-badge"
                        :class="'mi-badge-' + currentAlgorithm"
                        >{{
                            algoDisplay[currentAlgorithm] || currentAlgorithm
                        }}</span
                    >
                    <span class="mi-desc" :data-tip="algoTip">{{
                        algoShortDesc
                    }}</span>
                </div>

                <span class="sk">observation</span>
                <div class="sv mi-sv">
                    <span class="mi-val"
                        >{{ obsLabel }} · {{ colorLabel }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Input image resolution and format fed to the encoder. Smaller sizes train faster; grayscale reduces dimensionality; frame stacking provides temporal context so the agent can perceive motion."
                        >Input resolution and format</span
                    >
                </div>

                <span class="sk">obs width</span>
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="
                            openConfigEdit(
                                'observation',
                                'width',
                                mi.observation_width,
                                $event,
                            )
                        "
                        >{{ mi.observation_width ?? "—" }}</span
                    >
                </div>

                <span class="sk">obs height</span>
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="
                            openConfigEdit(
                                'observation',
                                'height',
                                mi.observation_height,
                                $event,
                            )
                        "
                        >{{ mi.observation_height ?? "—" }}</span
                    >
                </div>

                <span class="sk">grayscale</span>
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="
                            openConfigEdit(
                                'observation',
                                'grayscale',
                                mi.grayscale,
                                $event,
                            )
                        "
                        >{{ mi.grayscale ? "yes" : "no" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Converts RGB input to single-channel grayscale, reducing the observation tensor from 3 channels to 1. Lower memory and faster training, but loses color information some games rely on."
                        >1ch vs 3ch input</span
                    >
                </div>

                <span class="sk">frame stack</span>
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="
                            openConfigEdit(
                                'observation',
                                'frame_stack',
                                mi.frame_stack,
                                $event,
                            )
                        "
                        >{{ mi.frame_stack ?? "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Number of consecutive frames stacked together as a single observation. Enables the agent to infer velocity and direction of moving objects from pixel differences between frames."
                        >Frames per observation</span
                    >
                </div>

                <span class="sk">actions</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ actionLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Discrete keyboard actions the agent can take each step. Defined by the game's action space config (e.g. noop, left, right, fire for Breakout)."
                        >{{
                            ((mi.action_keys as string[]) || []).join(", ")
                        }}</span
                    >
                </div>

                <span class="sk">encoder</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ encoderLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Convolutional stack that converts raw pixel frames into a latent representation. Channel progression defines the depth and capacity of the feature extractor."
                        >Pixel-to-latent feature extractor</span
                    >
                </div>

                <span class="sk">latent dim</span>
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="
                            openConfigEdit(
                                'world_model',
                                'latent_dim',
                                mi.latent_dim,
                                $event,
                            )
                        "
                        >{{ mi.latent_dim ?? "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Size of the latent embedding vector produced by the encoder. Higher dimensions capture more detail but increase memory and compute. Typical range: 128–512."
                        >Latent space size</span
                    >
                </div>

                <span class="sk">predictor</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ predictorLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Transformer module that predicts future latent states given the current latent and a sequence of actions. Depth and heads control its capacity. Trained during JEPA world model pretraining."
                        >Action-conditioned future predictor</span
                    >
                </div>

                <span class="sk">q-network</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ qNetLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Fully-connected network that estimates action values (Q-values) from the latent representation. Dueling architecture separates state-value and advantage streams for more stable learning."
                        >Action value estimator</span
                    >
                </div>
            </div>

            <div class="pipe-nav">
                <button
                    @click="activeStep = nextStep(0)"
                    class="btn-tiny pipe-next"
                >
                    next →
                </button>
            </div>
        </div>

        <!-- Step 1: Collect Data -->
        <div v-show="activeStep === 1" class="pipe-panel pipe-panel-data">
            <div class="pipe-panel-desc">
                Run random episodes to collect experience data. The agent takes
                <strong>random actions</strong> and records transitions — used
                to train the JEPA world model and seed the replay buffer.
            </div>

            <div class="pipe-data-layout">
                <div class="pipe-data-fields">
                    <div class="settings-table">
                        <span class="sk" title="Random episodes to run"
                            >episodes</span
                        >
                        <div
                            class="sv clickable-field"
                            @click="openFieldEdit('collectEpisodes', $event)"
                        >
                            {{ collectEpisodes }}
                        </div>

                        <span class="sk" title="Max steps per episode"
                            >max steps</span
                        >
                        <div
                            class="sv clickable-field"
                            @click="openFieldEdit('collectMaxSteps', $event)"
                        >
                            {{ collectMaxSteps }}
                        </div>

                        <span class="sk" title="Auto-generated experiment name"
                            >run name</span
                        >
                        <div class="sv">
                            <span
                                style="color: var(--accent); font-size: 10px"
                                >{{ autoCollectName }}</span
                            >
                        </div>
                    </div>
                </div>

                <div class="pipe-data-datasets">
                    <div class="pipe-ds-header">Collected Datasets</div>
                    <VDropdown
                        v-model="selectedDataset"
                        :options="datasetOptions"
                        title="Previously collected random-action datasets"
                        full-width
                        compact
                    />
                    <div
                        v-if="runs.collectedDatasets.length === 0"
                        class="pipe-ds-empty"
                    >
                        No datasets yet. Collect data to create one.
                    </div>
                    <div v-else class="pipe-ds-list">
                        <div
                            v-for="d in runs.collectedDatasets.slice(0, 5)"
                            :key="d.name"
                            class="pipe-ds-item"
                            :class="{
                                'pipe-ds-selected': selectedDataset === d.name,
                            }"
                            @click="selectedDataset = d.name"
                        >
                            <span class="pipe-ds-name">{{ d.name }}</span>
                            <span class="pipe-ds-meta"
                                >{{ d.episodes }} eps · avg
                                {{ fmtNum(d.mean_score) }}</span
                            >
                        </div>
                    </div>
                </div>
            </div>

            <template v-if="training.isCollecting && training.collectJob">
                <div class="pipe-data-progress">
                    <span class="pipe-dp-label">Collecting</span>
                    <div class="pipe-dp-track">
                        <div
                            class="pipe-dp-fill"
                            :style="{
                                width:
                                    Math.min(
                                        100,
                                        (training.collectJob.episodes_done /
                                            training.collectJob
                                                .episodes_target) *
                                            100,
                                    ) + '%',
                            }"
                        ></div>
                    </div>
                    <span class="pipe-dp-count"
                        >{{ training.collectJob.episodes_done }}/{{
                            training.collectJob.episodes_target
                        }}</span
                    >
                    <span
                        v-if="training.collectJob.mean_score"
                        class="pipe-dp-score"
                        >avg {{ fmtNum(training.collectJob.mean_score) }}</span
                    >
                </div>
            </template>
            <div v-if="collectError" class="tc-error" style="font-size: 9px">
                {{ collectError }}
            </div>

            <div class="pipe-nav">
                <button @click="activeStep = prevStep(1)" class="btn-tiny">
                    ← back
                </button>
                <button
                    v-if="!training.isCollecting"
                    @click="startCollect"
                    class="btn-tiny btn-accent-tiny"
                    :disabled="busy"
                >
                    collect
                </button>
                <button
                    v-if="training.isCollecting"
                    @click="stopCollect"
                    class="btn-tiny"
                    style="color: var(--red)"
                >
                    stop
                </button>
                <button
                    @click="activeStep = nextStep(1)"
                    class="btn-tiny pipe-next"
                >
                    next →
                </button>
            </div>
        </div>

        <!-- Step 2: Pretrain JEPA -->
        <div v-show="activeStep === 2" class="pipe-panel pipe-panel-jepa">
            <template v-if="!isFrozenJepa">
                <div
                    class="pipe-panel-desc"
                    style="color: var(--muted); font-style: italic"
                >
                    Not needed for
                    {{ algoDisplay[currentAlgorithm] || currentAlgorithm }}.
                    Skip to Train.
                </div>
            </template>
            <template v-else>
                <div class="pipe-panel-desc">
                    Pretrain the JEPA world model on collected data. Learns to
                    predict future latent states from current observations and
                    actions.
                </div>

                <div class="settings-table">
                    <span class="sk" title="Gradient steps">steps</span>
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('worldSteps', $event)"
                    >
                        {{ worldSteps }}
                    </div>

                    <span class="sk" title="Batch size">batch</span>
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('worldBatch', $event)"
                    >
                        <span v-if="worldBatch">{{ worldBatch }}</span>
                        <span v-else class="cf-muted">auto</span>
                    </div>

                    <span class="sk" title="Learning rate">lr</span>
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('worldLr', $event)"
                    >
                        <span v-if="worldLr">{{ worldLr }}</span>
                        <span v-else class="cf-muted">auto</span>
                    </div>

                    <span class="sk" title="Pre-collect browser steps"
                        >pre-collect</span
                    >
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('worldCollectSteps', $event)"
                    >
                        <span v-if="worldCollectSteps">{{
                            worldCollectSteps
                        }}</span>
                        <span v-else class="cf-muted">auto</span>
                    </div>

                    <span class="sk" title="Log interval">log every</span>
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('worldDashEvery', $event)"
                    >
                        {{ worldDashEvery }}
                    </div>

                    <template
                        v-if="training.isWorldTraining && training.worldJob"
                    >
                        <span class="sk">run</span>
                        <span class="sv" style="color: var(--accent)">{{
                            training.worldJob.run_name
                        }}</span>
                    </template>
                    <div
                        v-if="worldError"
                        class="tc-error"
                        style="grid-column: 1/-1; font-size: 9px"
                    >
                        {{ worldError }}
                    </div>
                </div>
            </template>

            <div class="pipe-nav">
                <button @click="activeStep = prevStep(2)" class="btn-tiny">
                    ← back
                </button>
                <template v-if="isFrozenJepa">
                    <button
                        v-if="!training.isWorldTraining"
                        @click="startWorldTraining"
                        class="btn-tiny btn-accent-tiny"
                        :disabled="busy"
                    >
                        train world
                    </button>
                    <button
                        v-if="training.isWorldTraining"
                        @click="stopWorldTraining"
                        class="btn-tiny"
                        style="color: var(--red)"
                    >
                        stop
                    </button>
                </template>
                <button
                    @click="activeStep = nextStep(2)"
                    class="btn-tiny pipe-next"
                >
                    next →
                </button>
            </div>
        </div>

        <!-- Step 3: Train RL Agent -->
        <div v-show="activeStep === 3" class="pipe-panel pipe-panel-train">
            <div class="pipe-panel-desc">
                Train the RL agent. Starts by filling the replay buffer with
                random actions (<strong>warmup</strong>), then learns from
                experience.
            </div>

            <div class="settings-table">
                <template v-if="isFrozenJepa">
                    <span class="sk" title="JEPA encoder checkpoint"
                        >jepa ckpt</span
                    >
                    <div
                        class="sv clickable-field"
                        @click="openFieldEdit('jepaCheckpoint', $event)"
                    >
                        <span v-if="jepaCheckpoint">{{ jepaCheckpoint }}</span>
                        <span v-else class="cf-muted">none</span>
                    </div>
                </template>

                <template v-if="hasCheckpoint">
                    <span class="sk" title="Resume from checkpoint">from</span>
                    <VDropdown
                        v-model="selectedCheckpoint"
                        :options="checkpointOptions"
                        full-width
                        compact
                    />
                </template>

                <span class="sk" title="Random steps before learning begins"
                    >warmup</span
                >
                <div class="sv mi-sv">
                    <span
                        class="clickable-field"
                        @click="openFieldEdit('warmup', $event)"
                        >{{ warmup }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Random steps before learning begins. Fills the replay buffer with initial experience so the first gradient updates are stable. Set higher for complex environments."
                        >random steps to fill replay buffer</span
                    >
                </div>

                <span class="sk" title="Additional training steps"
                    >+ steps</span
                >
                <div class="sv">
                    <span
                        class="clickable-field"
                        @click="openFieldEdit('additionalSteps', $event)"
                        style="display: flex; align-items: center; gap: 4px"
                    >
                        {{ additionalSteps }}
                        <span
                            v-if="hasCheckpoint && checkpointStep() > 0"
                            style="color: var(--muted); font-size: 9px"
                            >→ {{ checkpointStep() + additionalSteps }}</span
                        >
                    </span>
                </div>

                <span class="sk" title="Minibatch size">batch</span>
                <div
                    class="sv clickable-field"
                    @click="openFieldEdit('batch', $event)"
                >
                    <span v-if="batch">{{ batch }}</span>
                    <span v-else class="cf-muted">auto</span>
                </div>

                <span class="sk" title="Learning rate override">lr</span>
                <div
                    class="sv clickable-field"
                    @click="openFieldEdit('lr', $event)"
                >
                    <span v-if="lr">{{ lr }}</span>
                    <span v-else class="cf-muted">auto</span>
                </div>

                <span class="sk" title="Log every N steps">log every</span>
                <div
                    class="sv clickable-field"
                    @click="openFieldEdit('logEvery', $event)"
                >
                    {{ logEvery }}
                </div>

                <span class="sk" title="Show browser during training"
                    >headed</span
                >
                <div class="sv mi-sv">
                    <input
                        type="checkbox"
                        v-model="headed"
                        :disabled="busy"
                        style="
                            margin: 0;
                            accent-color: var(--accent);
                            flex-shrink: 0;
                        "
                    />
                    <span
                        class="mi-desc"
                        data-tip="Show the browser window during training. Useful for visually debugging agent behavior, but significantly slower due to rendering overhead. Keep off for serious training runs."
                        >show browser window (slower)</span
                    >
                </div>
            </div>

            <div class="pipe-nav">
                <button @click="activeStep = prevStep(3)" class="btn-tiny">
                    ← back
                </button>
                <button
                    @click="watchAiPlay"
                    v-if="!busy && hasCheckpoint"
                    class="btn-accent-tiny"
                >
                    watch
                </button>
                <button
                    @click="startTraining"
                    v-if="!busy"
                    class="btn-accent-tiny"
                >
                    train
                </button>
                <button
                    @click="stopTraining"
                    v-if="busy"
                    class="btn-danger-tiny"
                >
                    stop
                </button>
            </div>
        </div>

        <!-- Field edit popover -->
        <div
            v-if="editingField"
            class="popover-backdrop"
            @click="closeFieldEdit"
        ></div>
        <div v-if="editingField" class="popover" :style="editPopoverStyle">
            <div class="popover-label">{{ editingField }}</div>
            <input
                v-model="editValue"
                type="text"
                class="popover-input"
                @keydown.enter="saveFieldEdit"
                @keydown.escape="closeFieldEdit"
                autofocus
            />
            <div class="popover-actions">
                <button class="btn-tiny" @click="closeFieldEdit">cancel</button>
                <button class="btn-accent-tiny" @click="saveFieldEdit">
                    save
                </button>
            </div>
        </div>

        <!-- Config field edit popover -->
        <div
            v-if="editingConfigField"
            class="popover-backdrop"
            @click="closeConfigFieldEdit"
        ></div>
        <div
            v-if="editingConfigField"
            class="popover"
            :style="configPopoverStyle"
        >
            <div class="popover-label">
                {{ configEditMeta.group }}.{{ configEditMeta.key }}
            </div>
            <input
                v-model="configEditValue"
                type="text"
                class="popover-input"
                @keydown.enter="saveConfigFieldEdit"
                @keydown.escape="closeConfigFieldEdit"
                autofocus
            />
            <div class="popover-actions">
                <button class="btn-tiny" @click="closeConfigFieldEdit">
                    cancel
                </button>
                <button class="btn-accent-tiny" @click="saveConfigFieldEdit">
                    save
                </button>
            </div>
        </div>

        <!-- New-run popover -->
        <div
            v-if="showNewRun"
            class="popover-backdrop"
            @click="showNewRun = false"
        ></div>
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
                <button @click="showNewRun = false" class="btn-tiny">
                    cancel
                </button>
                <button @click="confirmNewRun" class="btn-accent-tiny">
                    create
                </button>
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
.pipe-run-row .vdd {
    flex: 1;
    min-width: 0;
}
.pipe-smoke-lbl {
    display: flex;
    align-items: center;
    gap: 3px;
    font-size: 9px;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
}
.pipe-smoke-lbl input {
    accent-color: var(--accent);
    margin: 0;
}

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
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
}
.pipe-st-time {
    margin-left: auto;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--text);
}
.pipe-st-eta {
    font-family: "IBM Plex Mono", monospace;
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
    padding: 8px 8px 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: none;
    border-bottom: none;
}

.pipe-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    background: none;
    border: none;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 3px;
    flex-shrink: 0;
    transition: all 0.15s;
    min-width: 42px;
}
.pipe-node:hover {
    background: var(--surface-2);
}

.pipe-node-ic {
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    background: var(--bg);
    font-weight: 600;
    transition: all 0.15s;
    flex-shrink: 0;
}
.pipe-node-lb {
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    font-weight: 500;
    white-space: nowrap;
}

.pipe-node.ns-active .pipe-node-ic {
    border-color: var(--accent);
    color: var(--bg);
    background: var(--accent);
    font-weight: 700;
    box-shadow: 0 0 10px rgba(196, 145, 82, 0.45);
}
.pipe-node.ns-active .pipe-node-lb {
    color: var(--accent);
    font-weight: 700;
}

.pipe-node.ns-done .pipe-node-ic {
    border-color: var(--green);
    color: var(--bg);
    background: var(--green);
    font-weight: 700;
}
.pipe-node.ns-done .pipe-node-lb {
    color: var(--green);
    font-weight: 600;
}

.pipe-node.ns-running .pipe-node-ic {
    border-color: var(--green);
    color: var(--bg);
    background: var(--green);
    font-weight: 700;
    animation: pulse 1.5s infinite;
    box-shadow: 0 0 10px rgba(93, 158, 93, 0.4);
}

.pipe-node.ns-error .pipe-node-ic {
    border-color: var(--red);
    color: var(--red);
    background: rgba(185, 82, 76, 0.08);
}
.pipe-node.ns-error .pipe-node-lb {
    color: var(--red);
}

.pipe-node.ns-ready .pipe-node-ic {
    border-color: rgba(196, 145, 82, 0.5);
    color: var(--accent);
    background: rgba(196, 145, 82, 0.2);
    font-weight: 700;
}
.pipe-node.ns-ready .pipe-node-lb {
    color: var(--accent);
    font-weight: 600;
}

.pipe-node.ns-skipped {
    opacity: 0.4;
}
.pipe-node.ns-skipped .pipe-node-ic {
    border-style: dashed;
}

.pipe-conn {
    flex: 1;
    height: 2px;
    background: var(--border);
    margin-top: 12px;
    margin-left: 3px;
    margin-right: 3px;
    transition: background 0.2s;
    border-radius: 1px;
}
.pipe-conn.pipe-conn-done {
    background: rgba(93, 158, 93, 0.45);
}
.pipe-conn.pipe-conn-ready {
    background: rgba(196, 145, 82, 0.3);
}

/* Step panels */
.pipe-panel {
    padding: 12px 10px 8px;
    border-radius: 0 0 3px 3px;
    border: 1px solid var(--border);
    border-top: none;
    background: var(--bg);
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.pipe-panel-model {
    background: linear-gradient(
        180deg,
        rgba(78, 137, 186, 0.04) 0%,
        var(--bg) 100%
    );
}
.pipe-panel-data {
    background: linear-gradient(
        180deg,
        rgba(93, 158, 93, 0.04) 0%,
        var(--bg) 100%
    );
}
.pipe-panel-jepa {
    background: linear-gradient(
        180deg,
        rgba(196, 145, 82, 0.04) 0%,
        var(--bg) 100%
    );
}
.pipe-panel-train {
    background: linear-gradient(
        180deg,
        rgba(196, 145, 82, 0.03) 0%,
        var(--bg) 100%
    );
}

.pipe-panel-desc {
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
    padding: 0 2px;
}
.pipe-panel-desc strong {
    color: var(--text);
    font-weight: 600;
}

/* Validation error only */
.pipe-validation {
    display: flex;
    align-items: center;
}
.pipe-val-err {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--red);
    background: rgba(185, 82, 76, 0.08);
    border: 1px solid rgba(185, 82, 76, 0.2);
    border-radius: 3px;
    padding: 2px 8px;
}

/* Settings table — unified base (kept in sync with components.css base) */
.settings-table {
    display: grid;
    grid-template-columns: 68px 1fr;
    gap: 2px 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 5px 8px;
    font-size: 10px;
}
.settings-table .sk {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 8px;
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: flex;
    align-items: center;
    padding-top: 2px;
}
.settings-table .sv {
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    color: var(--text);
    min-width: 0;
    display: flex;
    align-items: baseline;
    gap: 8px;
}

/* Editable fields — > prefix + subtle styling */
.settings-table .sv :deep(.clickable-field),
.settings-table .sv.clickable-field {
    display: inline-flex;
    align-items: center;
    cursor: pointer;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    line-height: 1;
    color: var(--accent);
    font-weight: 500;
    transition: all 0.15s;
    border-color: var(--accent);
    border-bottom-style: solid;
    flex-shrink: 0;
}

/* Read-only values */
.settings-table .sv .mi-val {
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    color: var(--text);
    font-weight: 500;
    flex-shrink: 0;
}

/* Inline description with hover tooltip */
.mi-desc {
    font-size: 9px;
    color: var(--muted);
    position: relative;
    cursor: help;
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: 1.4;
}
.mi-desc::after {
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
    transition:
        opacity 0.15s,
        transform 0.15s;
    z-index: 100;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
}
.mi-desc:hover::after {
    opacity: 1;
    transform: translateY(0);
    pointer-events: auto;
}

/* Algorithm badge */
.mi-badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 2px;
    font-size: 10px;
    font-weight: 600;
    font-family: "IBM Plex Mono", monospace;
    flex-shrink: 0;
}
.mi-badge-dqn {
    background: rgba(93, 158, 93, 0.12);
    color: var(--green);
}
.mi-badge-frozen_jepa_dqn {
    background: rgba(78, 137, 186, 0.12);
    color: var(--blue);
}
.mi-badge-linear_q {
    background: rgba(191, 160, 62, 0.12);
    color: var(--yellow);
}
.mi-badge-joint_jepa_dqn {
    background: rgba(196, 145, 82, 0.12);
    color: var(--accent);
}

/* Muted placeholder */
.cf-muted {
    color: var(--muted);
    font-style: italic;
}

/* Data collection layout */
.pipe-data-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
}
@media (max-width: 600px) {
    .pipe-data-layout {
        grid-template-columns: 1fr;
    }
}
.pipe-data-fields .settings-table {
    border-radius: 3px;
}

/* Datasets panel */
.pipe-data-datasets {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.pipe-ds-header {
    font-size: 8px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}
.pipe-ds-empty {
    font-size: 9px;
    color: var(--muted);
    font-style: italic;
    padding: 4px 0;
}
.pipe-ds-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.pipe-ds-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 6px;
    border-radius: 2px;
    font-size: 10px;
    cursor: pointer;
    transition: background 0.12s;
    background: var(--surface);
    border: 1px solid var(--border);
}
.pipe-ds-item:hover {
    background: var(--surface-2);
}
.pipe-ds-item.pipe-ds-selected {
    border-color: rgba(93, 158, 93, 0.4);
    background: rgba(93, 158, 93, 0.06);
}
.pipe-ds-name {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.pipe-ds-meta {
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--muted);
    flex-shrink: 0;
    margin-left: 6px;
}

/* Collection progress */
.pipe-data-progress {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 6px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 3px;
}
.pipe-dp-label {
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--green);
    flex-shrink: 0;
}
.pipe-dp-track {
    flex: 1;
    height: 3px;
    background: var(--bg);
    border-radius: 2px;
    overflow: hidden;
}
.pipe-dp-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--green), var(--accent));
    border-radius: 2px;
    transition: width 0.3s ease;
}
.pipe-dp-count {
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
    color: var(--text);
    font-weight: 500;
    flex-shrink: 0;
}
.pipe-dp-score {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--accent);
    flex-shrink: 0;
}

/* Navigation */
.pipe-nav {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 4px;
    margin-top: 2px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
}
.pipe-next {
    color: var(--accent) !important;
    border-color: rgba(196, 145, 82, 0.3) !important;
}

.pipe-ii {
    border: none;
    background: transparent;
    color: var(--text);
    outline: none;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
}
.pipe-ii:focus {
    border-bottom: 1px solid var(--accent) !important;
}
.pipe-ii:hover {
    border-bottom: 1px solid var(--border) !important;
}

.pipe-lbl {
    display: flex;
    align-items: center;
    gap: 2px;
    color: var(--muted);
    font-size: 9px;
    font-family: "IBM Plex Mono", monospace;
}

@keyframes pulse {
    0%,
    100% {
        opacity: 1;
    }
    50% {
        opacity: 0.4;
    }
}
</style>
