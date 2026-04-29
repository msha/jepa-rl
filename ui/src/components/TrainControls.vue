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
const logEvery = ref(5);
const headed = ref(false);
const jepaCheckpoint = ref("");
const selectedCheckpoint = ref("");
const controlError = ref("");
const modelDraftDirty = ref(false);
const runAlgorithm = ref("dqn");
const runObsWidth = ref("");
const runObsHeight = ref("");
const runGrayscale = ref("false");
const runFrameStack = ref("");
const runLatentDim = ref("");
const runGamma = ref("");
const runBatchSize = ref("");
const runLearningStarts = ref("");
const runTrainEvery = ref("");
const runTargetUpdate = ref("");
const runAgentLr = ref("");
const runEpsilonStart = ref("");
const runEpsilonEnd = ref("");
const runEpsilonDecay = ref("");

const algorithmOptions = [
    { value: "dqn", label: "Pixel DQN" },
    { value: "frozen_jepa_dqn", label: "Frozen JEPA + DQN" },
    { value: "joint_jepa_dqn", label: "Joint JEPA + DQN" },
    { value: "linear_q", label: "Linear Q" },
];

const colorModeOptions = [
    { value: "false", label: "RGB" },
    { value: "true", label: "Grayscale" },
];

const pipeRunOptions = computed(() => [
    { value: "", label: "select run or create a new by starting from here..." },
    ...runs.runs.map((r) => ({ value: r.name, label: fmtRun(r) })),
]);

const checkpointOptions = computed(() =>
    availableCheckpoints.value.map((c) => ({ value: c.file, label: c.label })),
);

const activeModelInfo = computed<Record<string, unknown>>(() => {
    if (runs.selectedRun && runs.runModelInfo) return runs.runModelInfo;
    return training.baseModelInfo || training.modelInfo || {};
});

const hasPendingRun = computed(() => !!targetRunName.value && !runs.selectedRun);
const activeRunName = computed(() => runs.selectedRun || targetRunName.value);
const modelSettingsLocked = computed(() => !!runs.selectedRun || busy.value);
const canCreateRun = computed(
    () => hasPendingRun.value && !busy.value && !!targetRunName.value.trim(),
);

const currentAlgorithm = computed(() => {
    return runAlgorithm.value || String(activeModelInfo.value.algorithm || "dqn");
});
const isFrozenJepa = computed(
    () => currentAlgorithm.value === "frozen_jepa_dqn",
);
watch(isFrozenJepa, (frozen) => {
    if (!frozen && activeStep.value === 2) activeStep.value = 1;
});

watch(
    () => configStore.currentPath,
    () => {
        if (runs.selectedRun) return;
        modelDraftDirty.value = false;
        syncModelDraftFromConfig();
    },
);

function onNewRun(e: Event) {
    const detail = (e as CustomEvent).detail;
    if (!detail?.name) return;
    runs.clearSelection();
    targetRunName.value = detail.name;
    modelDraftDirty.value = false;
    syncModelDraftFromConfig();
    activeStep.value = 0;
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

const busy = computed(() => training.isTraining || training.isEvaluating || training.isCollecting);
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
    if (training.isCollecting) return "tc-dot running";
    if (props.job?.status === "error" || props.evalJob?.status === "error")
        return "tc-dot error";
    if (training.collectJob?.status === "error") return "tc-dot error";
    if (props.job?.status === "completed") return "tc-dot stopped";
    if (training.collectJob?.status === "completed") return "tc-dot stopped";
    return "tc-dot idle";
});

const statusText = computed(() => {
    if (isTraining.value) return "training";
    if (isEvaluating.value) return "evaluating";
    if (training.isCollecting) return "collecting";
    if (props.job?.status === "completed") return "completed";
    if (props.job?.status === "stopped") return "stopped";
    if (props.job?.status === "error") return "error";
    if (props.evalJob?.status === "error") return "eval error";
    if (training.collectJob?.status === "completed") return "collected";
    if (training.collectJob?.status === "error") return "collect error";
    return "idle";
});

const statusDetail = computed(() => {
    if (training.isCollecting && training.collectJob) {
        const cj = training.collectJob;
        const parts = [`ep ${cj.episodes_done}/${cj.episodes_target}`];
        if (cj.mean_score) parts.push(`avg ${fmtNum(cj.mean_score)}`);
        return parts.join(" · ");
    }
    if (training.collectJob?.status === "completed") {
        const cj = training.collectJob;
        return `${cj.episodes_done} eps · avg ${fmtNum(cj.mean_score)}`;
    }
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
    if (training.isCollecting && training.collectJob?.started_at) {
        const t = training.collectJob.started_at;
        return t > 1e12 ? t / 1000 : t;
    }
    if (!props.job?.started_at) return null;
    return props.job.started_at > 1e12
        ? props.job.started_at / 1000
        : props.job.started_at;
});

const elapsed = computed(() => {
    if (startTs.value == null) return null;
    if (!isTraining.value && !training.isCollecting) return null;
    return Math.max(0, now.value - Math.floor(startTs.value));
});

const currentStep = computed(() =>
    Number(props.latestStep?.step ?? props.summary?.steps ?? 0),
);

const eta = computed(() => {
    if (training.isCollecting && training.collectJob) {
        const done = training.collectJob.episodes_done;
        const target = training.collectJob.episodes_target;
        if (done <= 0 || elapsed.value == null) return null;
        const rate = done / elapsed.value;
        const remaining = target - done;
        if (remaining <= 0) return null;
        return remaining / rate;
    }
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
    if (training.isCollecting && training.collectJob) {
        const target = training.collectJob.episodes_target;
        if (!target) return null;
        return Math.min(1, training.collectJob.episodes_done / target);
    }
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

async function onRunChange() {
    targetRunName.value = "";
    await runs.loadRunDetail(runs.selectedRun);
    modelDraftDirty.value = false;
    syncModelDraftFromConfig();
    await training.refresh();
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
    runs.clearSelection();
    window.dispatchEvent(new CustomEvent("new-run", { detail: { name } }));
    targetRunName.value = name;
    showNewRun.value = false;
}

const availableCheckpoints = computed(() => {
    if (runs.selectedRun) return runs.checkpoints;
    if (runs.checkpoints.length) return runs.checkpoints;
    return training.runDir?.checkpoints ?? [];
});
const evalRunDir = computed(() =>
    runs.selectedRun ? runs.runDir : runs.runDir || training.runDir?.dir || "",
);
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
        logEvery.value = defaults.dashboard_every || 5;
    } catch {
        /* use defaults */
    }
});

function syncModelDraftFromConfig() {
    const mi = activeModelInfo.value;
    runAlgorithm.value = String(mi.algorithm || "dqn");
    runObsWidth.value = String(mi.observation_width ?? "");
    runObsHeight.value = String(mi.observation_height ?? "");
    runGrayscale.value = String(Boolean(mi.grayscale));
    runFrameStack.value = String(mi.frame_stack ?? "");
    runLatentDim.value = String(mi.latent_dim ?? "");
    runGamma.value = String(mi.gamma ?? "");
    runBatchSize.value = String(mi.batch_size ?? "");
    runLearningStarts.value = String(mi.learning_starts ?? "");
    runTrainEvery.value = String(mi.train_every ?? "");
    runTargetUpdate.value = String(mi.target_update_interval ?? "");
    runAgentLr.value = String(mi.agent_lr ?? "");
    runEpsilonStart.value = String(mi.epsilon_start ?? "");
    runEpsilonEnd.value = String(mi.epsilon_end ?? "");
    runEpsilonDecay.value = String(mi.epsilon_decay_steps ?? "");
}

watch(
    activeModelInfo,
    () => {
        if (!modelDraftDirty.value) syncModelDraftFromConfig();
    },
    { immediate: true, deep: true },
);

watch(
    () => runs.selectedRun,
    (name) => {
        if (name) targetRunName.value = "";
        modelDraftDirty.value = false;
        syncModelDraftFromConfig();
    },
);

function markModelDraftDirty() {
    modelDraftDirty.value = true;
}

function buildRunOverrides(): { group: string; key: string; value: string }[] {
    return [
        { group: "agent", key: "algorithm", value: runAlgorithm.value },
        { group: "agent", key: "gamma", value: runGamma.value },
        { group: "agent", key: "batch_size", value: runBatchSize.value },
        { group: "agent", key: "learning_starts", value: runLearningStarts.value },
        { group: "agent", key: "train_every", value: runTrainEvery.value },
        { group: "agent", key: "target_update_interval", value: runTargetUpdate.value },
        { group: "agent", key: "optimizer.lr", value: runAgentLr.value },
        { group: "observation", key: "width", value: runObsWidth.value },
        { group: "observation", key: "height", value: runObsHeight.value },
        { group: "observation", key: "grayscale", value: runGrayscale.value },
        { group: "observation", key: "frame_stack", value: runFrameStack.value },
        { group: "world_model", key: "latent_dim", value: runLatentDim.value },
        { group: "exploration", key: "epsilon_start", value: runEpsilonStart.value },
        { group: "exploration", key: "epsilon_end", value: runEpsilonEnd.value },
        { group: "exploration", key: "epsilon_decay_steps", value: runEpsilonDecay.value },
    ].filter((item) => item.value !== "");
}

async function createTrainingRun() {
    controlError.value = "";
    const name = targetRunName.value.trim();
    if (!name) {
        controlError.value = "enter a run name first";
        return;
    }
    try {
        await runs.createRun(name, buildRunOverrides());
        targetRunName.value = "";
        modelDraftDirty.value = false;
        await training.refresh();
        syncModelDraftFromConfig();
    } catch (e) {
        controlError.value = e instanceof Error ? e.message : String(e);
        console.error(e);
    }
}

async function startTraining() {
    controlError.value = "";
    const experiment = runs.selectedRun || "";
    if (!experiment) {
        controlError.value = "create or select a run first";
        return;
    }
    const ckptStep = checkpointStep();
    const totalSteps = ckptStep + Number(additionalSteps.value);
    const payload: Record<string, unknown> = {
        experiment,
        steps: totalSteps,
        dashboard_every: Number(logEvery.value) || 5,
        headed: headed.value,
    };
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
const saveFrames = ref(false);


const selectedDatasetInfo = computed(() => {
    if (!selectedDataset.value) return null;
    return runs.collectedDatasets.find((d) => d.name === selectedDataset.value) ?? null;
});

function fmtBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes}b`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}kb`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}mb`;
}

async function deleteDataset(name: string) {
    if (!confirm(`Delete dataset "${name}"?`)) return;
    try {
        await runs.deleteDataset(name);
        if (selectedDataset.value === name) selectedDataset.value = "";
    } catch (e) {
        collectError.value = e instanceof Error ? e.message : String(e);
    }
}

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

// Auto-generate collection name: <game>-rand-act-<next_free_int>
const autoCollectName = computed(() => {
    const game = training.gameName || "game";
    const existingNames = new Set(runs.collectedDatasets.map((d) => d.name));
    let n = 1;
    while (existingNames.has(`${game}-rand-act-${n}`)) n++;
    return `${game}-rand-act-${n}`;
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
            save_frames: saveFrames.value,
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
    if (!runs.selectedRun) {
        worldError.value = "create or select a run first";
        return;
    }
    try {
        await training.startWorldTraining({
            experiment: `${runs.selectedRun}_world`,
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
const editLabels: Record<string, string> = {
    collectEpisodes: "episodes",
    collectMaxSteps: "max steps",
    worldSteps: "world steps",
    worldBatch: "world batch",
    worldLr: "world lr",
    worldCollectSteps: "pre-collect",
    worldDashEvery: "world log every",
    additionalSteps: "training steps",
    logEvery: "log every",
    jepaCheckpoint: "JEPA checkpoint",
    runObsWidth: "observation width",
    runObsHeight: "observation height",
    runFrameStack: "frame stack",
    runLatentDim: "latent dimension",
    runGamma: "discount gamma",
    runBatchSize: "run batch",
    runLearningStarts: "learning starts",
    runTrainEvery: "train every",
    runTargetUpdate: "target sync",
    runAgentLr: "policy learning rate",
    runEpsilonStart: "epsilon start",
    runEpsilonEnd: "epsilon end",
    runEpsilonDecay: "epsilon decay steps",
};
const editingFieldLabel = computed(() =>
    editingField.value ? editLabels[editingField.value] || editingField.value : "",
);

const modelFieldKeys = new Set([
    "runObsWidth",
    "runObsHeight",
    "runFrameStack",
    "runLatentDim",
    "runGamma",
    "runBatchSize",
    "runLearningStarts",
    "runTrainEvery",
    "runTargetUpdate",
    "runAgentLr",
    "runEpsilonStart",
    "runEpsilonEnd",
    "runEpsilonDecay",
]);

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
        case "additionalSteps":
            return String(additionalSteps.value);
        case "logEvery":
            return String(logEvery.value);
        case "jepaCheckpoint":
            return jepaCheckpoint.value;
        case "runObsWidth":
            return runObsWidth.value;
        case "runObsHeight":
            return runObsHeight.value;
        case "runFrameStack":
            return runFrameStack.value;
        case "runLatentDim":
            return runLatentDim.value;
        case "runGamma":
            return runGamma.value;
        case "runBatchSize":
            return runBatchSize.value;
        case "runLearningStarts":
            return runLearningStarts.value;
        case "runTrainEvery":
            return runTrainEvery.value;
        case "runTargetUpdate":
            return runTargetUpdate.value;
        case "runAgentLr":
            return runAgentLr.value;
        case "runEpsilonStart":
            return runEpsilonStart.value;
        case "runEpsilonEnd":
            return runEpsilonEnd.value;
        case "runEpsilonDecay":
            return runEpsilonDecay.value;
        default:
            return "";
    }
}

function fieldCls(key: string): Record<string, boolean> {
    const locked = modelFieldKeys.has(key) && modelSettingsLocked.value;
    return { "clickable-field": !locked, "cf-locked": locked };
}

function openFieldEdit(key: string, event: MouseEvent) {
    if (modelFieldKeys.has(key) && modelSettingsLocked.value) return;
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
    if (modelFieldKeys.has(key) && modelSettingsLocked.value) {
        editingField.value = null;
        return;
    }
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
        case "additionalSteps":
            additionalSteps.value = n;
            break;
        case "logEvery":
            logEvery.value = n;
            break;
        case "jepaCheckpoint":
            jepaCheckpoint.value = val;
            break;
        case "runObsWidth":
            runObsWidth.value = val;
            markModelDraftDirty();
            break;
        case "runObsHeight":
            runObsHeight.value = val;
            markModelDraftDirty();
            break;
        case "runFrameStack":
            runFrameStack.value = val;
            markModelDraftDirty();
            break;
        case "runLatentDim":
            runLatentDim.value = val;
            markModelDraftDirty();
            break;
        case "runGamma":
            runGamma.value = val;
            markModelDraftDirty();
            break;
        case "runBatchSize":
            runBatchSize.value = val;
            markModelDraftDirty();
            break;
        case "runLearningStarts":
            runLearningStarts.value = val;
            markModelDraftDirty();
            break;
        case "runTrainEvery":
            runTrainEvery.value = val;
            markModelDraftDirty();
            break;
        case "runTargetUpdate":
            runTargetUpdate.value = val;
            markModelDraftDirty();
            break;
        case "runAgentLr":
            runAgentLr.value = val;
            markModelDraftDirty();
            break;
        case "runEpsilonStart":
            runEpsilonStart.value = val;
            markModelDraftDirty();
            break;
        case "runEpsilonEnd":
            runEpsilonEnd.value = val;
            markModelDraftDirty();
            break;
        case "runEpsilonDecay":
            runEpsilonDecay.value = val;
            markModelDraftDirty();
            break;
    }
    editingField.value = null;
}

function closeFieldEdit() {
    editingField.value = null;
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
    const s0: StepStatus = runs.selectedRun
        ? "done"
        : targetRunName.value
          ? "ready"
          : "pending";
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
            : runs.selectedRun
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
    const vi = visibleSteps.value.indexOf(idx);
    return String(vi >= 0 ? vi + 1 : idx + 1);
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
const mi = activeModelInfo;
const obsLabel = computed(() => {
    const w = runObsWidth.value || mi.value.observation_width || "?";
    const h = runObsHeight.value || mi.value.observation_height || "?";
    return `${w}×${h}`;
});
const colorLabel = computed(() => {
    const gray = runGrayscale.value === "true";
    const fs = runFrameStack.value || mi.value.frame_stack || "?";
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
                @click="openNewRun"
                class="btn-tiny"
                :disabled="busy"
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
            <template v-if="elapsed != null">
                <span class="pipe-st-time">{{ fmtDuration(elapsed) }}</span>
                <span v-if="eta != null" class="pipe-st-eta"
                    >~{{ fmtDuration(eta) }}</span
                >
            </template>
            <button v-if="isTraining || isEvaluating" @click="stopTraining" class="btn-danger-tiny">
                stop
            </button>
            <button v-if="training.isCollecting" @click="stopCollect" class="btn-danger-tiny">
                stop
            </button>
        </div>
        <div v-if="progress != null" class="pipe-prog">
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
                Configure model settings while this run is still a draft. Create
                the run to write the config snapshot; after that, model settings
                are locked and training can only use the snapshot.
            </div>

            <div class="pipe-run-card">
                <span class="sk">run</span>
                <span class="sv">
                    <span class="mi-val">{{ activeRunName || "new run needed" }}</span>
                    <span class="run-lock-state">
                        {{ runs.selectedRun ? "locked snapshot" : hasPendingRun ? "draft" : "not created" }}
                    </span>
                </span>
            </div>

            <!-- Config validation error only -->
            <div
                v-if="validationResult && !validationResult.ok"
                class="pipe-validation"
            >
                <span class="pipe-val-err">✗ {{ validationResult.error }}</span>
            </div>

            <!-- Model info as settings table -->
            <div class="settings-table mi-table" :class="{ 'mi-table-locked': modelSettingsLocked }">
                <span class="sk">algorithm</span>
                <div class="sv mi-sv">
                    <VDropdown
                        v-model="runAlgorithm"
                        :options="algorithmOptions"
                        compact
                        :disabled="modelSettingsLocked"
                        @change="markModelDraftDirty"
                    />
                    <span
                        class="mi-desc"
                        :data-tip="algoTip"
                        :data-short="algoShortDesc"
                    ></span>
                </div>

                <span class="sk">observation</span>
                <div class="sv mi-sv">
                    <span class="mi-val"
                        >{{ obsLabel }} · {{ colorLabel }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Input image resolution and format fed to the encoder. Smaller sizes train faster; grayscale reduces dimensionality; frame stacking provides temporal context so the agent can perceive motion."
                        data-short="Input resolution and format"
                    ></span>
                </div>

                <span class="sk">obs width</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runObsWidth')"
                        @click="openFieldEdit('runObsWidth', $event)"
                        >{{ runObsWidth || "—" }}</span
                    >
                </div>

                <span class="sk">obs height</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runObsHeight')"
                        @click="openFieldEdit('runObsHeight', $event)"
                        >{{ runObsHeight || "—" }}</span
                    >
                </div>

                <span class="sk">color</span>
                <div class="sv mi-sv">
                    <VDropdown
                        v-model="runGrayscale"
                        :options="colorModeOptions"
                        compact
                        :disabled="modelSettingsLocked"
                        @change="markModelDraftDirty"
                    />
                    <span
                        class="mi-desc"
                        data-tip="Converts RGB input to single-channel grayscale, reducing the observation tensor from 3 channels to 1. Lower memory and faster training, but loses color information some games rely on."
                        data-short="1ch vs 3ch input"
                    ></span>
                </div>

                <span class="sk">frame stack</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runFrameStack')"
                        @click="openFieldEdit('runFrameStack', $event)"
                        >{{ runFrameStack || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Number of consecutive frames stacked together as a single observation. Enables the agent to infer velocity and direction of moving objects from pixel differences between frames."
                        data-short="Frames per observation"
                    ></span>
                </div>

                <span class="sk">actions</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ actionLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Discrete keyboard actions the agent can take each step. Defined by the game's action space config (e.g. noop, left, right, fire for Breakout)."
                        :data-short="
                            ((mi.action_keys as string[]) || []).join(', ')
                        "
                    ></span>
                </div>

                <span class="sk">encoder</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ encoderLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Convolutional stack that converts raw pixel frames into a latent representation. Channel progression defines the depth and capacity of the feature extractor."
                        data-short="Pixel-to-latent feature extractor"
                    ></span>
                </div>

                <span class="sk">latent dim</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runLatentDim')"
                        @click="openFieldEdit('runLatentDim', $event)"
                        >{{ runLatentDim || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Size of the latent embedding vector produced by the encoder. Higher dimensions capture more detail but increase memory and compute. Typical range: 128–512."
                        data-short="Latent space size"
                    ></span>
                </div>

                <span class="sk">predictor</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ predictorLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Transformer module that predicts future latent states given the current latent and a sequence of actions. Depth and heads control its capacity. Trained during JEPA world model pretraining."
                        data-short="Action-conditioned future predictor"
                    ></span>
                </div>

                <span class="sk">q-network</span>
                <div class="sv mi-sv">
                    <span class="mi-val">{{ qNetLabel }}</span>
                    <span
                        class="mi-desc"
                        data-tip="Fully-connected network that estimates action values (Q-values) from the latent representation. Dueling architecture separates state-value and advantage streams for more stable learning."
                        data-short="Action value estimator"
                    ></span>
                </div>

                <span class="sk">gamma</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runGamma')"
                        @click="openFieldEdit('runGamma', $event)"
                        >{{ runGamma || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Reward discount factor. Higher values make the policy care more about delayed rewards."
                        data-short="future reward weighting"
                    ></span>
                </div>

                <span class="sk">batch</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runBatchSize')"
                        @click="openFieldEdit('runBatchSize', $event)"
                        >{{ runBatchSize || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Replay minibatch size used for policy updates. Saved into the run snapshot and locked once the run is created."
                        data-short="policy update minibatch"
                    ></span>
                </div>

                <span class="sk">learn starts</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runLearningStarts')"
                        @click="openFieldEdit('runLearningStarts', $event)"
                        >{{ runLearningStarts || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Environment steps collected before gradient updates begin. Saved into the run snapshot and locked once the run is created."
                        data-short="replay warmup before learning"
                    ></span>
                </div>

                <span class="sk">train every</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runTrainEvery')"
                        @click="openFieldEdit('runTrainEvery', $event)"
                        >{{ runTrainEvery || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Run policy gradient updates every N environment steps."
                        data-short="policy update cadence"
                    ></span>
                </div>

                <span class="sk">target sync</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runTargetUpdate')"
                        @click="openFieldEdit('runTargetUpdate', $event)"
                        >{{ runTargetUpdate || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Step interval for copying the online Q-network into the target Q-network."
                        data-short="target network interval"
                    ></span>
                </div>

                <span class="sk">policy lr</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runAgentLr')"
                        @click="openFieldEdit('runAgentLr', $event)"
                        >{{ runAgentLr || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Learning rate for the policy optimizer. This belongs to the run snapshot and cannot be changed after the run is created."
                        data-short="policy optimizer step size"
                    ></span>
                </div>

                <span class="sk">epsilon</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runEpsilonStart')"
                        @click="openFieldEdit('runEpsilonStart', $event)"
                        >{{ runEpsilonStart || "—" }}</span
                    >
                    <span>→</span>
                    <span
                        :class="fieldCls('runEpsilonEnd')"
                        @click="openFieldEdit('runEpsilonEnd', $event)"
                        >{{ runEpsilonEnd || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Epsilon-greedy exploration schedule. The agent starts random and decays toward greedy action selection."
                        data-short="exploration start and floor"
                    ></span>
                </div>

                <span class="sk">eps decay</span>
                <div class="sv mi-sv">
                    <span
                        :class="fieldCls('runEpsilonDecay')"
                        @click="openFieldEdit('runEpsilonDecay', $event)"
                        >{{ runEpsilonDecay || "—" }}</span
                    >
                    <span
                        class="mi-desc"
                        data-tip="Number of environment steps over which epsilon decays from start to end."
                        data-short="exploration decay horizon"
                    ></span>
                </div>
            </div>

            <div class="pipe-nav">
                <button
                    v-if="hasPendingRun"
                    @click="createTrainingRun"
                    class="btn-accent-tiny"
                    :disabled="!canCreateRun"
                >
                    create run
                </button>
                <button
                    @click="activeStep = nextStep(0)"
                    class="btn-tiny pipe-next"
                    :disabled="!runs.selectedRun"
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
                        <div class="sv">
                            <span class="clickable-field" @click="openFieldEdit('collectEpisodes', $event)">{{ collectEpisodes }}</span>
                        </div>

                        <span class="sk" title="Max steps per episode"
                            >max steps</span
                        >
                        <div class="sv">
                            <span class="clickable-field" @click="openFieldEdit('collectMaxSteps', $event)">{{ collectMaxSteps }}</span>
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
                        <span class="sk">screenshots</span>
                        <label class="pipe-data-toggle">
                            <input type="checkbox" v-model="saveFrames" />
                            <span class="toggle-track">
                                <span class="toggle-thumb"></span>
                            </span>
                            <span class="toggle-label">{{ saveFrames ? 'on' : 'off' }}</span>
                        </label>
                    </div>
                </div>

                <div class="pipe-data-datasets">
                    <div class="pipe-ds-header">
                        Datasets
                        <span class="pipe-ds-count">{{ runs.collectedDatasets.length }}</span>
                    </div>
                    <div
                        v-if="runs.collectedDatasets.length === 0"
                        class="pipe-ds-empty"
                    >
                        No datasets yet. Collect data to create one.
                    </div>
                    <template v-else>
                        <VDropdown
                            v-model="selectedDataset"
                            :options="datasetOptions"
                            title="Select a collected dataset"
                            full-width
                            compact
                        />
                        <button
                            v-if="selectedDataset"
                            class="btn-danger-tiny"
                            style="margin-top: 2px; align-self: flex-end"
                            @click="deleteDataset(selectedDataset)"
                        >delete dataset</button>
                    </template>
                </div>
            </div>

            <!-- Dataset detail (always rendered, fixed height) -->
            <div class="pipe-ds-detail" :class="{ 'pipe-ds-detail-active': !!selectedDatasetInfo }">
                <template v-if="selectedDatasetInfo">
                    <div class="pipe-ds-detail-row">
                        <span class="pipe-ds-detail-name">{{ selectedDatasetInfo.name }}</span>
                    </div>
                    <div class="pipe-ds-detail-stats">
                        <span>{{ selectedDatasetInfo.episodes }} eps</span>
                        <span>{{ selectedDatasetInfo.total_steps }} steps</span>
                        <span>~{{ fmtNum(selectedDatasetInfo.mean_length) }} len</span>
                        <span>{{ fmtBytes(selectedDatasetInfo.size_bytes) }}</span>
                    </div>
                    <div class="pipe-ds-detail-scores">
                        <span>avg {{ fmtNum(selectedDatasetInfo.mean_score) }}</span>
                        <span>med {{ fmtNum(selectedDatasetInfo.median_score) }}</span>
                        <span>best {{ fmtNum(selectedDatasetInfo.max_score) }}</span>
                        <span>min {{ fmtNum(selectedDatasetInfo.min_score) }}</span>
                    </div>
                </template>
                <div v-else class="pipe-ds-detail-empty">select a dataset to view details</div>
            </div>

            <div v-if="collectError" class="tc-error pipe-err">
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
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('worldSteps', $event)">{{ worldSteps }}</span>
                    </div>

                    <span class="sk" title="Batch size">batch</span>
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('worldBatch', $event)">
                            <span v-if="worldBatch">{{ worldBatch }}</span>
                            <span v-else class="cf-muted">auto</span>
                        </span>
                    </div>

                    <span class="sk" title="Learning rate">lr</span>
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('worldLr', $event)">
                            <span v-if="worldLr">{{ worldLr }}</span>
                            <span v-else class="cf-muted">auto</span>
                        </span>
                    </div>

                    <span class="sk" title="Pre-collect browser steps"
                        >pre-collect</span
                    >
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('worldCollectSteps', $event)">
                            <span v-if="worldCollectSteps">{{ worldCollectSteps }}</span>
                            <span v-else class="cf-muted">auto</span>
                        </span>
                    </div>

                    <span class="sk" title="Log interval">log every</span>
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('worldDashEvery', $event)">{{ worldDashEvery }}</span>
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
                        :disabled="busy || !runs.selectedRun"
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
                Train or resume the selected run. Model, optimizer, replay, and
                exploration settings come from the locked run snapshot.
            </div>

            <div class="settings-table">
                <template v-if="isFrozenJepa">
                    <span class="sk" title="JEPA encoder checkpoint"
                        >jepa ckpt</span
                    >
                    <div class="sv">
                        <span class="clickable-field" @click="openFieldEdit('jepaCheckpoint', $event)">
                            <span v-if="jepaCheckpoint">{{ jepaCheckpoint }}</span>
                            <span v-else class="cf-muted">none</span>
                        </span>
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

                <span class="sk" title="Log every N steps">log every</span>
                <div class="sv">
                    <span class="clickable-field" @click="openFieldEdit('logEvery', $event)">{{ logEvery }}</span>
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
                        data-short="show browser window (slower)"
                    ></span>
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
                    :disabled="!runs.selectedRun"
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
            <div class="popover-label">{{ editingFieldLabel }}</div>
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

.pipe-run-card {
    display: grid;
    grid-template-columns: 68px 1fr;
    gap: 2px 10px;
    align-items: center;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 5px 8px;
    font-size: 10px;
}
.pipe-run-card .sk {
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 8px;
}
.pipe-run-card .sv {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
}
.run-lock-state {
    color: var(--muted);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.mi-table-locked .clickable-field {
    color: var(--text);
    border-bottom-color: transparent;
    cursor: default;
}
.mi-table-locked .clickable-field:hover {
    color: var(--text);
    border-bottom-color: transparent;
}
.cf-locked {
    color: var(--text);
    cursor: default;
    border-bottom: none;
}
.cf-locked:hover {
    color: var(--text);
    border-bottom: none;
}

button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
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
    position: relative;
    cursor: help;
    flex: 1;
    min-width: 0;
    line-height: 1.4;
    display: flex;
    font-size: 0;
}
.mi-desc::before {
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
.pipe-data-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
}
.pipe-data-toggle input {
    display: none;
}
.toggle-track {
    width: 28px;
    height: 14px;
    background: var(--border);
    border-radius: 7px;
    position: relative;
    transition: background 0.2s;
}
.pipe-data-toggle input:checked + .toggle-track {
    background: var(--accent);
}
.toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--text);
    transition: transform 0.2s;
}
.pipe-data-toggle input:checked + .toggle-track .toggle-thumb {
    transform: translateX(14px);
}
.toggle-label {
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
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
    display: flex;
    align-items: center;
    gap: 6px;
}
.pipe-ds-count {
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--accent);
    background: rgba(196, 145, 82, 0.1);
    border-radius: 2px;
    padding: 0 4px;
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
.pipe-ds-main {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
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
}
.pipe-ds-del {
    background: none;
    border: none;
    color: var(--muted);
    font-size: 13px;
    cursor: pointer;
    padding: 0 2px;
    line-height: 1;
    opacity: 0;
    transition: opacity 0.15s, color 0.15s;
}
.pipe-ds-item:hover .pipe-ds-del,
.pipe-ds-detail .pipe-ds-del {
    opacity: 1;
}
.pipe-ds-del:hover {
    color: var(--red);
}

/* Selected dataset detail */
.pipe-ds-detail {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 5px 8px;
    display: flex;
    flex-direction: column;
    gap: 3px;
    min-height: 52px;
    justify-content: center;
}
.pipe-ds-detail-empty {
    font-size: 9px;
    color: var(--muted);
    font-style: italic;
    opacity: 0.5;
    text-align: center;
}
.pipe-ds-detail-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.pipe-ds-detail-name {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    font-weight: 600;
    color: var(--green);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.pipe-ds-detail-stats,
.pipe-ds-detail-scores {
    display: flex;
    gap: 8px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--muted);
}
.pipe-ds-detail-scores span:first-child {
    color: var(--accent);
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
