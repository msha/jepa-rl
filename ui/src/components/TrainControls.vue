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
// Extended settings (wizard step 2)
const runPredictorDepth = ref("");
const runPredictorHeads = ref("");
const runReplayCapacity = ref("");
const runReplaySeqLen = ref("");
const runReplayPrioritized = ref("false");
const runGameActionRepeat = ref("");
const runGameFps = ref("");
const runGameMaxSteps = ref("");
const runRewardPatienceSteps = ref("");
const runRewardPenalty = ref("");

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

const jepaCkptOptions = computed(() => {
    const ckpts = training.jepaCheckpoints;
    return [
        { value: "", label: "auto (latest)" },
        ...ckpts.map((c) => ({ value: c.file, label: c.label })),
    ];
});

const activeModelInfo = computed<Record<string, unknown>>(() => {
    if (runs.selectedRun && runs.runModelInfo) return runs.runModelInfo;
    return training.baseModelInfo || training.modelInfo || {};
});

const hasPendingRun = computed(
    () => !!targetRunName.value && !runs.selectedRun,
);
const activeRunName = computed(() => runs.selectedRun || targetRunName.value);
const modelSettingsLocked = computed(() => !!runs.selectedRun || busy.value);
const canCreateRun = computed(
    () => hasPendingRun.value && !busy.value && !!targetRunName.value.trim(),
);

const currentAlgorithm = computed(() => {
    return (
        runAlgorithm.value || String(activeModelInfo.value.algorithm || "dqn")
    );
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

const busy = computed(
    () =>
        training.isTraining ||
        training.isEvaluating ||
        training.isCollecting ||
        training.isWorldTraining,
);
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
    if (training.isWorldTraining) return "tc-dot running";
    if (training.isCollecting) return "tc-dot running";
    if (props.job?.status === "error" || props.evalJob?.status === "error")
        return "tc-dot error";
    if (training.worldJob?.status === "error") return "tc-dot error";
    if (training.collectJob?.status === "error") return "tc-dot error";
    if (props.job?.status === "completed") return "tc-dot stopped";
    if (training.worldJob?.status === "completed") return "tc-dot stopped";
    if (training.collectJob?.status === "completed") return "tc-dot stopped";
    return "tc-dot idle";
});

const statusText = computed(() => {
    if (isTraining.value) return "training";
    if (isEvaluating.value) return "evaluating";
    if (training.isWorldTraining) {
        const ls = props.latestStep || {};
        if (ls.phase === "collecting") return "collecting";
        return "world training";
    }
    if (training.isCollecting) return "collecting";
    if (props.job?.status === "completed") return "completed";
    if (props.job?.status === "stopped") return "stopped";
    if (props.job?.status === "error") return "error";
    if (training.worldJob?.status === "completed") return "world done";
    if (training.worldJob?.status === "error") return "world error";
    if (props.evalJob?.status === "error") return "eval error";
    if (training.collectJob?.status === "completed") return "collected";
    if (training.collectJob?.status === "error") return "collect error";
    return "idle";
});

const statusDetail = computed(() => {
    if (training.isWorldTraining && training.worldJob) {
        const wj = training.worldJob;
        const ls = props.latestStep || {};
        if (ls.phase === "collecting") {
            const done = ls.collect_step ?? 0;
            const total = ls.collect_total ?? 0;
            const eps = ls.episodes ?? 0;
            const parts = [`collect ${done}/${total}`];
            if (eps) parts.push(`${eps} eps`);
            return parts.join(" · ");
        }
        const trainStep =
            typeof ls.step === "number" && ls.phase !== "collecting"
                ? ls.step
                : 0;
        const parts = [`step ${trainStep}/${wj.requested_steps}`];
        if (ls.loss != null) parts.push(`loss ${fmtNum(ls.loss)}`);
        return parts.join(" · ");
    }
    if (training.worldJob?.status === "completed") {
        return `${training.worldJob.requested_steps} steps`;
    }
    if (training.worldJob?.status === "error") {
        return training.worldJob.error || "training failed";
    }
    if (training.isCollecting && training.collectJob) {
        const cj = training.collectJob;
        const parts = [`ep ${cj.episodes_done}/${cj.episodes_target}`];
        if (cj.avg_steps) parts.push(`~${Math.round(cj.avg_steps)} steps/ep`);
        if (cj.mean_score) parts.push(`avg ${fmtNum(cj.mean_score)}`);
        const elapsed = (Date.now() / 1000) - (cj.started_at || Date.now() / 1000);
        if (cj.total_steps > 0 && elapsed > 2) {
            const rate = cj.total_steps / elapsed;
            const remaining = cj.episodes_target - cj.episodes_done;
            if (remaining > 0 && cj.avg_steps > 0) {
                const etaSec = (remaining * cj.avg_steps) / rate;
                parts.push(`~${fmtDuration(etaSec)}`);
            }
            parts.push(`${fmtNum(rate)} steps/s`);
        }
        return parts.join(" · ");
    }
    if (training.collectJob?.status === "completed") {
        const cj = training.collectJob;
        const parts = [`${cj.episodes_done} eps`];
        if (cj.total_steps) parts.push(`${cj.total_steps} steps`);
        parts.push(`avg ${fmtNum(cj.mean_score)}`);
        return parts.join(" · ");
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
    if (training.isWorldTraining && training.worldJob?.started_at) {
        const t = training.worldJob.started_at;
        return t > 1e12 ? t / 1000 : t;
    }
    if (!props.job?.started_at) return null;
    return props.job.started_at > 1e12
        ? props.job.started_at / 1000
        : props.job.started_at;
});

const elapsed = computed(() => {
    if (startTs.value == null) return null;
    if (
        !isTraining.value &&
        !training.isCollecting &&
        !training.isWorldTraining
    )
        return null;
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
    if (training.isWorldTraining && training.worldJob) {
        const ls = props.latestStep || {};
        if (ls.phase === "collecting") {
            const total = Number(ls.collect_total ?? 0);
            if (!total) return 0;
            return Math.min(1, Number(ls.collect_step ?? 0) / total);
        }
        const trainStep =
            typeof ls.step === "number" && ls.phase !== "collecting"
                ? ls.step
                : 0;
        return Math.min(1, trainStep / training.worldJob.requested_steps);
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
    const parts: string[] = [];
    if (r.experiment_name && r.experiment_name !== r.name) {
        parts.push(r.experiment_name, r.name);
    } else {
        parts.push(r.name);
    }
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

// === WIZARD ===
// Step order: 0=Algorithm  1=Settings+Name
const wizardOpen = ref(false);
const wizardStep = ref(0);
const wizardName = ref("");
const wizardNameManuallyEdited = ref(false);
const expandedAlgo = ref<string | null>("dqn");

const algoNameMap: Record<string, string[]> = {
    dqn: ["dqn", "pixel_dqn", "PixelDQN"],
    frozen_jepa_dqn: ["frozen_jepa_dqn", "FrozenJEPADQN"],
    joint_jepa_dqn: ["joint_jepa_dqn", "JointJEPADQN"],
    linear_q: ["linear_q", "LinearQ"],
};

const algoShortName: Record<string, string> = {
    dqn: "dqn",
    frozen_jepa_dqn: "fjepa",
    joint_jepa_dqn: "jjepa",
    linear_q: "linq",
};

function algoStats(algo: string) {
    const names = algoNameMap[algo] || [algo];
    const matching = runs.runs.filter((r) => names.includes(r.algorithm || ""));
    if (!matching.length) return null;
    return {
        count: matching.length,
        bestScore: Math.max(...matching.map((r) => r.best_score ?? 0)),
        totalSteps: matching.reduce((s, r) => s + (r.steps ?? 0), 0),
    };
}

function fmtSteps(n: number): string {
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
    if (n >= 1_000) return Math.round(n / 1_000) + "k";
    return String(n);
}

const isWizardJepa = computed(
    () =>
        runAlgorithm.value === "frozen_jepa_dqn" ||
        runAlgorithm.value === "joint_jepa_dqn",
);

const autoWizardName = computed(() => {
    const game = (training.gameName || "game")
        .toLowerCase()
        .replace(/\s+/g, "-");
    const algo = algoShortName[runAlgorithm.value] || runAlgorithm.value;
    const names = algoNameMap[runAlgorithm.value] || [runAlgorithm.value];
    const n =
        runs.runs.filter((r) => names.includes(r.algorithm || "")).length + 1;
    return `${game}-${algo}-r${n}`;
});

watch(autoWizardName, (name) => {
    if (!wizardNameManuallyEdited.value) wizardName.value = name;
});

// @ts-expect-error used in template
function onWizardNameInput() {
    wizardNameManuallyEdited.value = wizardName.value !== autoWizardName.value;
}
// @ts-expect-error used in template
function resetWizardName() {
    wizardNameManuallyEdited.value = false;
    wizardName.value = autoWizardName.value;
}

const algoCards = [
    {
        value: "dqn",
        label: "Pixel DQN",
        tag: "baseline",
        color: "#5d9e5d",
        summary: "End-to-end Q-learning directly from raw pixels.",
        bullets: [
            "Convolutional encoder trained from scratch",
            "Dueling architecture + Double DQN for stability",
            "Epsilon-greedy exploration that decays over time",
        ],
        rec: "Starting out or establishing a baseline score",
    },
    {
        value: "frozen_jepa_dqn",
        label: "Frozen JEPA + DQN",
        tag: "sample-efficient",
        color: "#4e89ba",
        summary:
            "Pretrain a world model encoder, then train only a Q-head on frozen latents.",
        bullets: [
            "Encoder learns environment dynamics offline first",
            "Q-head trains on rich, stable latent representations",
            "Typically reaches DQN scores in far fewer env steps",
        ],
        rec: "Maximising sample efficiency over pixel DQN",
    },
    {
        value: "joint_jepa_dqn",
        label: "Joint JEPA + DQN",
        tag: "experimental",
        color: "#c49152",
        summary:
            "Trains the JEPA encoder and DQN head jointly with shared gradients.",
        bullets: [
            "Single training loop for representation + control",
            "Encoder shaped by both world modeling and reward signals",
            "More complex, potentially more expressive than frozen",
        ],
        rec: "Exploring representation-RL co-training",
    },
    {
        value: "linear_q",
        label: "Linear Q",
        tag: "smoke test",
        color: "#7d7870",
        summary: "Trivial linear approximator for pipeline validation only.",
        bullets: [
            "No neural network — a single linear layer",
            "Validates the full training loop end-to-end cheaply",
            "Not intended for real gameplay",
        ],
        rec: "Debugging the pipeline, not real training",
    },
];

function selectAlgo(value: string) {
    runAlgorithm.value = value;
    markModelDraftDirty();
    expandedAlgo.value = expandedAlgo.value === value ? null : value;
}

function openWizard() {
    wizardStep.value = 0;
    expandedAlgo.value = runAlgorithm.value;
    syncModelDraftFromConfig();
    wizardNameManuallyEdited.value = false;
    wizardName.value = autoWizardName.value;
    wizardOpen.value = true;
}

function cancelWizard() {
    wizardOpen.value = false;
    syncModelDraftFromConfig();
}

function wizardNext() {
    if (wizardStep.value < 1) wizardStep.value = 1;
}

function wizardBack() {
    if (wizardStep.value > 0) wizardStep.value = 0;
}

async function confirmWizard() {
    const name = wizardName.value.trim();
    if (!name) return;
    runs.clearSelection();
    targetRunName.value = name;
    wizardOpen.value = false;
    await createTrainingRun();
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
    runPredictorDepth.value = String(mi.predictor_depth ?? "");
    runPredictorHeads.value = String(mi.predictor_heads ?? "");
    // extended fields default to empty (use config defaults)
    runReplayCapacity.value = "";
    runReplaySeqLen.value = "";
    runReplayPrioritized.value = "false";
    runGameActionRepeat.value = "";
    runGameFps.value = "";
    runGameMaxSteps.value = "";
    runRewardPatienceSteps.value = "";
    runRewardPenalty.value = "";
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
        {
            group: "agent",
            key: "learning_starts",
            value: runLearningStarts.value,
        },
        { group: "agent", key: "train_every", value: runTrainEvery.value },
        {
            group: "agent",
            key: "target_update_interval",
            value: runTargetUpdate.value,
        },
        { group: "agent", key: "optimizer.lr", value: runAgentLr.value },
        { group: "observation", key: "width", value: runObsWidth.value },
        { group: "observation", key: "height", value: runObsHeight.value },
        { group: "observation", key: "grayscale", value: runGrayscale.value },
        {
            group: "observation",
            key: "frame_stack",
            value: runFrameStack.value,
        },
        { group: "world_model", key: "latent_dim", value: runLatentDim.value },
        {
            group: "world_model",
            key: "predictor.depth",
            value: runPredictorDepth.value,
        },
        {
            group: "world_model",
            key: "predictor.heads",
            value: runPredictorHeads.value,
        },
        {
            group: "exploration",
            key: "epsilon_start",
            value: runEpsilonStart.value,
        },
        {
            group: "exploration",
            key: "epsilon_end",
            value: runEpsilonEnd.value,
        },
        {
            group: "exploration",
            key: "epsilon_decay_steps",
            value: runEpsilonDecay.value,
        },
        { group: "replay", key: "capacity", value: runReplayCapacity.value },
        {
            group: "replay",
            key: "sequence_length",
            value: runReplaySeqLen.value,
        },
        {
            group: "replay",
            key: "prioritized",
            value: runReplayPrioritized.value === "true" ? "true" : "",
        },
        {
            group: "game",
            key: "action_repeat",
            value: runGameActionRepeat.value,
        },
        { group: "game", key: "fps", value: runGameFps.value },
        {
            group: "game",
            key: "max_steps_per_episode",
            value: runGameMaxSteps.value,
        },
        {
            group: "reward",
            key: "zero_score_patience_steps",
            value: runRewardPatienceSteps.value,
        },
        {
            group: "reward",
            key: "zero_score_penalty",
            value: runRewardPenalty.value,
        },
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

const collectEpisodes = ref(Number(localStorage.getItem("jeparl_collectEpisodes") || 5));
const collectMaxSteps = ref(Number(localStorage.getItem("jeparl_collectMaxSteps") || 200));
const collectError = ref("");
const saveFrames = ref(localStorage.getItem("jeparl_saveFrames") === "true");
const dataMode = ref<"new" | "continue">("new");

// Persist settings
watch(collectEpisodes, v => localStorage.setItem("jeparl_collectEpisodes", String(v)));
watch(collectMaxSteps, v => localStorage.setItem("jeparl_collectMaxSteps", String(v)));
watch(saveFrames, v => localStorage.setItem("jeparl_saveFrames", String(v)));

const selectedDataset = ref("");

const selectedDatasetInfo = computed(() => {
    if (!selectedDataset.value) return null;
    return (
        runs.collectedDatasets.find((d) => d.name === selectedDataset.value) ??
        null
    );
});

const datasetOptions = computed(() => {
    const currentGame = training.gameName;
    const filtered = currentGame
        ? runs.collectedDatasets.filter((d) => d.game === currentGame)
        : runs.collectedDatasets;
    const opts = [{ value: "", label: "none selected" }];
    for (const d of filtered) {
        const parts = [d.name];
        if (d.episodes) parts.push(`${d.episodes} eps`);
        if (d.mean_score != null) parts.push(`avg ${fmtNum(d.mean_score)}`);
        opts.push({ value: d.name, label: parts.join(" · ") });
    }
    return opts;
});

// Auto-select best dataset for current game
watch(
    [() => runs.collectedDatasets, () => training.gameName],
    ([datasets, game]) => {
        const gameDatasets = game
            ? datasets.filter((d) => d.game === game)
            : datasets;
        // Keep current if it still exists and matches game
        if (selectedDataset.value) {
            if (gameDatasets.some((d) => d.name === selectedDataset.value)) return;
        }
        if (gameDatasets.length > 0) {
            const best = gameDatasets.reduce((a, b) =>
                (a.mean_score ?? -Infinity) >= (b.mean_score ?? -Infinity)
                    ? a
                    : b,
            );
            selectedDataset.value = best.name;
        } else {
            selectedDataset.value = "";
        }
    },
    { immediate: true },
);

// Reload datasets when collection finishes
watch(
    () => training.collectJob?.status,
    (status, prev) => {
        if (prev === "running" && (status === "completed" || status === "stopped")) {
            runs.loadCollectedDatasets();
        }
    },
);

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

async function openFolder(path: string) {
    try {
        await api("/api/open-folder", { path });
    } catch { /* swallow */ }
}

const openSettingSections = ref(
    new Set(["observation", "agent", "world", "exploration", "replay"]),
);
function toggleSettingSection(key: string) {
    if (openSettingSections.value.has(key))
        openSettingSections.value.delete(key);
    else openSettingSections.value.add(key);
}

// Auto-generate collection name: <game>-rand-act-<next_free_int>
const autoCollectName = computed(() => {
    const game = training.gameName || "game";
    const existingNames = new Set(runs.collectedDatasets.map((d) => d.name));
    let n = 1;
    while (existingNames.has(`${game}-rand-act-${n}`)) n++;
    return `${game}-rand-act-${n}`;
});

async function startCollect() {
    collectError.value = "";
    try {
        if (dataMode.value === "continue" && selectedDataset.value) {
            await training.startCollect({
                experiment: selectedDataset.value,
                episodes: collectEpisodes.value,
                max_steps: collectMaxSteps.value,
                headed: headed.value,
                save_frames: saveFrames.value,
                existing: true,
            });
        } else {
            const experiment = autoCollectName.value;
            await training.startCollect({
                experiment,
                episodes: collectEpisodes.value,
                max_steps: collectMaxSteps.value,
                headed: headed.value,
                save_frames: saveFrames.value,
            });
        }
        await runs.loadCollectedDatasets();
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
const worldCollectSteps = ref("2000");
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

// World training live stats (read from latestStep)
const worldPhase = computed(() => {
    const ls = props.latestStep || {};
    return String(ls.phase || (typeof ls.step === "number" && Number(ls.step) > 0 && ls.phase !== "collecting" ? "training" : ""));
});
const worldCollectDone = computed(() => Number((props.latestStep || {}).collect_step ?? 0));
const worldCollectTotal = computed(() => Number((props.latestStep || {}).collect_total ?? 0));
const worldEpisodes = computed(() => Number((props.latestStep || {}).episodes ?? 0));
const worldTrainStep = computed(() => {
    const ls = props.latestStep || {};
    return worldPhase.value === "training" && typeof ls.step === "number" ? ls.step : 0;
});
const worldTrainLoss = computed(() => {
    const ls = props.latestStep || {};
    return typeof ls.loss === "number" ? ls.loss : null;
});

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
    editingField.value
        ? editLabels[editingField.value] || editingField.value
        : "",
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
        : selectedDatasetInfo.value
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
        <!-- ========= WIZARD ========= -->
        <div v-if="wizardOpen" class="wiz">
            <!-- Breadcrumb -->
            <div class="wiz-crumb">
                <button
                    v-for="(crumb, i) in [
                        { label: 'Algorithm' },
                        { label: 'Settings' },
                        { label: 'Name' },
                    ]"
                    :key="i"
                    class="wiz-crumb-step"
                    :class="{
                        'wcs-active': wizardStep === i,
                        'wcs-done': wizardStep > i,
                    }"
                    @click="wizardStep > i && (wizardStep = i as 0 | 1 | 2)"
                >
                    <span class="wcs-num">{{
                        wizardStep > i ? "✓" : i + 1
                    }}</span>
                    <span class="wcs-lbl">{{ crumb.label }}</span>
                </button>
            </div>

            <!-- Step 0: Algorithm -->
            <div v-if="wizardStep === 0" class="wiz-body">
                <div class="wiz-step-header">
                    <div class="wiz-step-title">Choose an algorithm</div>
                    <div class="wiz-step-subtitle">
                        Click to select. Expand for details and run history.
                    </div>
                </div>
                <div class="wiz-algo-list">
                    <div
                        v-for="card in algoCards"
                        :key="card.value"
                        class="wac"
                        :class="{
                            'wac-selected': runAlgorithm === card.value,
                            'wac-expanded': expandedAlgo === card.value,
                        }"
                        :style="{ '--cc': card.color }"
                    >
                        <!-- Card row (always visible) -->
                        <div class="wac-row" @click="selectAlgo(card.value)">
                            <div class="wac-row-left">
                                <span class="wac-sel-dot"></span>
                                <span class="wac-name">{{ card.label }}</span>
                                <span
                                    class="wac-badge"
                                    :style="{
                                        background: card.color + '20',
                                        color: card.color,
                                        borderColor: card.color + '50',
                                    }"
                                    >{{ card.tag }}</span
                                >
                            </div>
                            <div class="wac-row-right">
                                <template v-if="algoStats(card.value)">
                                    <span class="wac-stat wac-stat-best">{{
                                        fmtNum(algoStats(card.value)!.bestScore)
                                    }}</span>
                                    <span class="wac-stat-sep">best</span>
                                    <span class="wac-stat"
                                        >{{
                                            algoStats(card.value)!.count
                                        }}r</span
                                    >
                                </template>
                                <span v-else class="wac-stat-none">—</span>
                                <button
                                    class="wac-toggle"
                                    @click.stop="
                                        expandedAlgo =
                                            expandedAlgo === card.value
                                                ? null
                                                : card.value
                                    "
                                    :title="
                                        expandedAlgo === card.value
                                            ? 'Collapse'
                                            : 'Expand'
                                    "
                                >
                                    {{
                                        expandedAlgo === card.value ? "▴" : "▾"
                                    }}
                                </button>
                            </div>
                        </div>
                        <!-- Expanded body -->
                        <div
                            v-if="expandedAlgo === card.value"
                            class="wac-body"
                        >
                            <p class="wac-summary">{{ card.summary }}</p>
                            <ul class="wac-bullets">
                                <li v-for="b in card.bullets" :key="b">
                                    {{ b }}
                                </li>
                            </ul>
                            <div class="wac-meta">
                                <span class="wac-rec"
                                    >Best for: {{ card.rec }}</span
                                >
                                <div
                                    class="wac-history"
                                    v-if="algoStats(card.value)"
                                >
                                    <span
                                        >{{
                                            algoStats(card.value)!.count
                                        }}
                                        run{{
                                            algoStats(card.value)!.count === 1
                                                ? ""
                                                : "s"
                                        }}</span
                                    >
                                    <span class="wac-stat-sep">·</span>
                                    <span class="wac-stat-best"
                                        >best
                                        {{
                                            fmtNum(
                                                algoStats(card.value)!
                                                    .bestScore,
                                            )
                                        }}</span
                                    >
                                    <span class="wac-stat-sep">·</span>
                                    <span
                                        >{{
                                            fmtSteps(
                                                algoStats(card.value)!
                                                    .totalSteps,
                                            )
                                        }}
                                        steps trained</span
                                    >
                                </div>
                                <div class="wac-history wac-no-history" v-else>
                                    no runs yet — be the first
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- LLM advisor hook -->
                <div class="wiz-advisor">
                    <span class="wiz-advisor-icon">✦</span>
                    <div class="wiz-advisor-text">
                        <span class="wiz-advisor-title"
                            >AI Hyperparameter Advisor</span
                        >
                        <span class="wiz-advisor-sub"
                            >Analyzes your run history to suggest optimal
                            settings</span
                        >
                    </div>
                    <button
                        class="wiz-advisor-btn"
                        disabled
                        title="Coming soon — will analyze your run history and suggest optimal hyperparameters"
                    >
                        Suggest settings
                    </button>
                </div>
            </div>

            <!-- Step 1: Settings -->
            <div
                v-else-if="wizardStep === 1"
                class="wiz-body wiz-body-settings"
            >
                <div class="wiz-step-header">
                    <div class="wiz-step-title">Configure settings</div>
                    <div class="wiz-step-subtitle">
                        Locked once the run is created. Blank fields use config
                        defaults.
                    </div>
                </div>

                <!-- Observation -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('observation')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("observation") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Observation</span>
                        <span class="wss-hint"
                            >{{ runObsWidth || "?" }}×{{
                                runObsHeight || "?"
                            }}
                            · {{ runGrayscale === "true" ? "gray" : "rgb" }} ·
                            {{ runFrameStack || "?" }}f</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('observation')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">width</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runObsWidth"
                                @input="markModelDraftDirty"
                                placeholder="160"
                                min="64"
                                max="640"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">height</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runObsHeight"
                                @input="markModelDraftDirty"
                                placeholder="120"
                                min="48"
                                max="480"
                        /></label>
                        <label class="wsf">
                            <span class="wsf-l">color</span>
                            <select
                                class="wsf-i wsf-sel"
                                v-model="runGrayscale"
                                @change="markModelDraftDirty"
                            >
                                <option value="false">RGB (3ch)</option>
                                <option value="true">Grayscale (1ch)</option>
                            </select>
                        </label>
                        <label class="wsf"
                            ><span class="wsf-l">frame stack</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runFrameStack"
                                @input="markModelDraftDirty"
                                placeholder="4"
                                min="1"
                                max="16"
                        /></label>
                    </div>
                </div>

                <!-- Agent -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('agent')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("agent") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Agent</span>
                        <span class="wss-hint"
                            >γ={{ runGamma || "0.997" }} · lr={{
                                runAgentLr || "1e-4"
                            }}
                            · batch={{ runBatchSize || "256" }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('agent')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">gamma</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runGamma"
                                @input="markModelDraftDirty"
                                placeholder="0.997"
                                min="0.9"
                                max="1"
                                step="0.001"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">batch size</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runBatchSize"
                                @input="markModelDraftDirty"
                                placeholder="256"
                                min="16"
                                max="2048"
                                step="16"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">learn starts</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runLearningStarts"
                                @input="markModelDraftDirty"
                                placeholder="10000"
                                min="100"
                                max="100000"
                                step="100"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">policy lr</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runAgentLr"
                                @input="markModelDraftDirty"
                                placeholder="0.0001"
                                min="0.00001"
                                max="0.01"
                                step="0.00001"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">train every</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runTrainEvery"
                                @input="markModelDraftDirty"
                                placeholder="4"
                                min="1"
                                max="32"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">target sync</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runTargetUpdate"
                                @input="markModelDraftDirty"
                                placeholder="2000"
                                min="100"
                                max="50000"
                                step="100"
                        /></label>
                    </div>
                </div>

                <!-- Exploration -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('exploration')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("exploration") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Exploration</span>
                        <span class="wss-hint"
                            >ε {{ runEpsilonStart || "1.0" }} →
                            {{ runEpsilonEnd || "0.05" }} over
                            {{
                                runEpsilonDecay
                                    ? fmtSteps(Number(runEpsilonDecay))
                                    : "500k"
                            }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('exploration')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">ε start</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runEpsilonStart"
                                @input="markModelDraftDirty"
                                placeholder="1.0"
                                min="0"
                                max="1"
                                step="0.01"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">ε end</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runEpsilonEnd"
                                @input="markModelDraftDirty"
                                placeholder="0.05"
                                min="0"
                                max="1"
                                step="0.01"
                        /></label>
                        <label class="wsf wsf-wide"
                            ><span class="wsf-l">decay steps</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runEpsilonDecay"
                                @input="markModelDraftDirty"
                                placeholder="500000"
                                min="1000"
                                max="5000000"
                                step="1000"
                        /></label>
                    </div>
                </div>

                <!-- World Model (JEPA only) -->
                <div class="wss" v-if="isWizardJepa">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('world')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("world") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">World Model</span>
                        <span class="wss-hint"
                            >latent={{ runLatentDim || "512" }} · depth={{
                                runPredictorDepth || "4"
                            }}
                            · heads={{ runPredictorHeads || "8" }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('world')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">latent dim</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runLatentDim"
                                @input="markModelDraftDirty"
                                placeholder="512"
                                min="64"
                                max="2048"
                                step="64"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">predictor depth</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runPredictorDepth"
                                @input="markModelDraftDirty"
                                placeholder="4"
                                min="1"
                                max="16"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">attn heads</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runPredictorHeads"
                                @input="markModelDraftDirty"
                                placeholder="8"
                                min="1"
                                max="32"
                        /></label>
                    </div>
                </div>

                <!-- Replay -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('replay')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("replay") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Replay Buffer</span>
                        <span class="wss-hint"
                            >capacity={{
                                runReplayCapacity
                                    ? fmtSteps(Number(runReplayCapacity))
                                    : "1M"
                            }}
                            · seq={{ runReplaySeqLen || "16" }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('replay')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">capacity</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runReplayCapacity"
                                @input="markModelDraftDirty"
                                placeholder="1000000"
                                min="1000"
                                max="10000000"
                                step="1000"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">seq length</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runReplaySeqLen"
                                @input="markModelDraftDirty"
                                placeholder="16"
                                min="1"
                                max="128"
                        /></label>
                        <label class="wsf wsf-wide">
                            <span class="wsf-l">prioritized</span>
                            <label class="wsf-toggle">
                                <input
                                    type="checkbox"
                                    :checked="runReplayPrioritized === 'true'"
                                    @change="
                                        runReplayPrioritized = (
                                            $event.target as HTMLInputElement
                                        ).checked
                                            ? 'true'
                                            : 'false';
                                        markModelDraftDirty();
                                    "
                                />
                                <span class="wsf-toggle-track"
                                    ><span class="wsf-toggle-thumb"></span
                                ></span>
                                <span class="wsf-toggle-lbl">{{
                                    runReplayPrioritized === "true"
                                        ? "on"
                                        : "off"
                                }}</span>
                            </label>
                        </label>
                    </div>
                </div>

                <!-- Game -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('game')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("game") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Game</span>
                        <span class="wss-hint"
                            >repeat={{ runGameActionRepeat || "4" }} · fps={{
                                runGameFps || "30"
                            }}
                            · max={{ runGameMaxSteps || "1000" }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('game')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">action repeat</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runGameActionRepeat"
                                @input="markModelDraftDirty"
                                placeholder="4"
                                min="1"
                                max="16"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">fps</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runGameFps"
                                @input="markModelDraftDirty"
                                placeholder="30"
                                min="10"
                                max="120"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">max steps/ep</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runGameMaxSteps"
                                @input="markModelDraftDirty"
                                placeholder="1000"
                                min="100"
                                max="10000"
                                step="100"
                        /></label>
                    </div>
                </div>

                <!-- Reward -->
                <div class="wss">
                    <button
                        class="wss-head"
                        @click="toggleSettingSection('reward')"
                    >
                        <span class="wss-arrow">{{
                            openSettingSections.has("reward") ? "▾" : "▸"
                        }}</span>
                        <span class="wss-title">Reward</span>
                        <span class="wss-hint"
                            >patience={{ runRewardPatienceSteps || "120" }} ·
                            penalty={{ runRewardPenalty || "0.01" }}</span
                        >
                    </button>
                    <div
                        v-if="openSettingSections.has('reward')"
                        class="wss-body"
                    >
                        <label class="wsf"
                            ><span class="wsf-l">0-score patience</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runRewardPatienceSteps"
                                @input="markModelDraftDirty"
                                placeholder="120"
                                min="0"
                                max="1000"
                        /></label>
                        <label class="wsf"
                            ><span class="wsf-l">0-score penalty</span
                            ><input
                                type="number"
                                class="wsf-i"
                                v-model="runRewardPenalty"
                                @input="markModelDraftDirty"
                                placeholder="0.01"
                                min="0"
                                max="1"
                                step="0.001"
                        /></label>
                    </div>
                </div>
            </div>

            <!-- Step 2: Name -->
            <div v-else class="wiz-body wiz-body-name">
                <div class="wiz-step-header">
                    <div class="wiz-step-title">Name this run</div>
                    <div class="wiz-step-subtitle">
                        Auto-generated from game + algorithm. Edit freely.
                    </div>
                </div>
                <div class="wiz-name-preview">
                    <span
                        class="wnp-algo"
                        :style="{
                            color: algoCards.find(
                                (c) => c.value === runAlgorithm,
                            )?.color,
                        }"
                    >
                        {{
                            algoCards.find((c) => c.value === runAlgorithm)
                                ?.label
                        }}
                    </span>
                    <span class="wnp-sep">·</span>
                    <span class="wnp-game">{{
                        training.gameName || "game"
                    }}</span>
                </div>
                <input
                    v-model="wizardName"
                    class="wiz-name-input"
                    placeholder="run name..."
                    @keydown.enter="confirmWizard"
                    @keydown.escape="cancelWizard"
                />
                <div class="wiz-name-hint">
                    <span class="wnh-key">↵ Enter</span> to create ·
                    <span class="wnh-auto" @click="wizardName = autoWizardName"
                        >↺ reset to auto</span
                    >
                </div>
            </div>

            <!-- Wizard footer -->
            <div class="wiz-footer">
                <button class="wiz-cancel" @click="cancelWizard">
                    ✕ cancel
                </button>
                <div class="wiz-nav">
                    <button
                        v-if="wizardStep > 0"
                        class="wiz-back"
                        @click="wizardBack"
                    >
                        ← back
                    </button>
                    <button
                        v-if="wizardStep < 2"
                        class="wiz-next"
                        @click="wizardNext"
                    >
                        next →
                    </button>
                    <button
                        v-if="wizardStep === 2"
                        class="wiz-create"
                        @click="confirmWizard"
                        :disabled="!wizardName.trim() || busy"
                    >
                        Create "{{ wizardName.trim() || "..." }}"
                    </button>
                </div>
            </div>
        </div>
        <!-- ========= END WIZARD ========= -->

        <!-- Run selector row (hidden when wizard is open) -->
        <template v-if="!wizardOpen">
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
                <button @click="openWizard" class="btn-tiny" :disabled="busy">
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
                <button
                    v-if="isTraining || isEvaluating"
                    @click="stopTraining"
                    class="btn-danger-tiny"
                >
                    stop
                </button>
                <button
                    v-if="training.isCollecting"
                    @click="stopCollect"
                    class="btn-danger-tiny"
                >
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
                    Configure model settings while this run is still a draft.
                    Create the run to write the config snapshot; after that,
                    model settings are locked and training can only use the
                    snapshot.
                </div>

                <div class="pipe-run-card">
                    <span class="sk">run</span>
                    <span class="sv">
                        <span class="mi-val">{{
                            activeRunName || "new run needed"
                        }}</span>
                        <span class="run-lock-state">
                            {{
                                runs.selectedRun
                                    ? "locked snapshot"
                                    : hasPendingRun
                                      ? "draft"
                                      : "not created"
                            }}
                        </span>
                    </span>
                </div>

                <!-- Config validation error only -->
                <div
                    v-if="validationResult && !validationResult.ok"
                    class="pipe-validation"
                >
                    <span class="pipe-val-err"
                        >✗ {{ validationResult.error }}</span
                    >
                </div>

                <!-- Model info as settings table -->
                <div
                    class="settings-table mi-table"
                    :class="{ 'mi-table-locked': modelSettingsLocked }"
                >
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
                    Collect experience data by running
                    <strong>random-action episodes</strong>. Each episode starts
                    from the game's initial state and runs until the agent runs
                    out of lives or hits the step limit. The resulting dataset
                    trains the JEPA world model and seeds the replay buffer.
                </div>

                <!-- Dataset selector -->
                <div class="settings-table mi-table">
                    <span class="sk">dataset</span>
                    <div class="sv mi-sv">
                        <VDropdown
                            v-if="datasetOptions.length > 1"
                            v-model="selectedDataset"
                            :options="datasetOptions"
                            compact
                            style="min-width: 180px"
                        />
                        <span v-else class="mi-val">none collected yet</span>
                        <button
                            v-if="!training.isCollecting"
                            @click="
                                dataMode = 'new';
                                startCollect();
                            "
                            class="btn-tiny btn-accent-tiny pipe-ds-action"
                            :disabled="busy"
                        >
                            new dataset
                        </button>
                        <button
                            v-if="!training.isCollecting && selectedDataset"
                            @click="
                                dataMode = 'continue';
                                startCollect();
                            "
                            class="btn-tiny pipe-ds-action"
                            :disabled="busy"
                        >
                            continue
                        </button>
                        <button
                            v-if="selectedDataset && !training.isCollecting"
                            class="btn-tiny pipe-ds-delete"
                            @click="deleteDataset(selectedDataset)"
                        >
                            delete
                        </button>
                        <button
                            v-if="training.isCollecting"
                            @click="stopCollect"
                            class="btn-tiny btn-danger-tiny pipe-ds-action"
                        >
                            stop
                        </button>
                    </div>

                    <span class="sk">name</span>
                    <div class="sv mi-sv">
                        <span class="mi-val">{{
                            dataMode === "continue" && selectedDataset
                                ? selectedDataset
                                : autoCollectName
                        }}</span>
                    </div>

                    <span class="sk">game</span>
                    <div class="sv mi-sv">
                        <span class="mi-val">{{ training.gameName || "—" }}</span>
                    </div>

                    <template v-if="selectedDatasetInfo">
                        <span class="sk">episodes</span>
                        <div class="sv mi-sv">
                            <span class="mi-val">{{
                                selectedDatasetInfo.episodes
                            }}</span>
                            <span
                                class="mi-val"
                                style="color: var(--muted); margin-left: 4px"
                                >{{
                                    selectedDatasetInfo.total_steps
                                }}
                                steps</span
                            >
                            <span
                                class="mi-desc"
                                data-tip="Completed episodes and total steps. One step = one action held for action_repeat frames (~133ms)."
                                data-short="episodes and total steps"
                            ></span>
                        </div>

                        <span class="sk">avg steps/ep</span>
                        <div class="sv mi-sv">
                            <span class="mi-val"
                                >~{{
                                    selectedDatasetInfo.total_steps &&
                                    selectedDatasetInfo.episodes
                                        ? Math.round(
                                              selectedDatasetInfo.total_steps /
                                                  selectedDatasetInfo.episodes,
                                          )
                                        : "—"
                                }}</span
                            >
                            <span
                                class="mi-val"
                                style="color: var(--muted); margin-left: 4px"
                                >{{
                                    fmtBytes(selectedDatasetInfo.size_bytes)
                                }}</span
                            >
                            <span
                                class="mi-desc"
                                data-tip="Average steps per episode. Shorter episodes usually mean the agent died faster. ~200 steps ≈ 27 seconds of game time."
                                data-short="avg steps per episode"
                            ></span>
                        </div>

                        <span class="sk">avg score</span>
                        <div class="sv mi-sv">
                            <span class="mi-val">{{
                                fmtNum(selectedDatasetInfo.mean_score)
                            }}</span>
                            <span
                                class="mi-val"
                                style="color: var(--muted); margin-left: 4px"
                                >med
                                {{ fmtNum(selectedDatasetInfo.median_score) }} ·
                                best
                                {{
                                    fmtNum(selectedDatasetInfo.max_score)
                                }}</span
                            >
                        </div>

                        <template v-if="selectedDatasetInfo.images_count">
                            <span class="sk">images</span>
                            <div class="sv mi-sv">
                                <span
                                    class="clickable-field"
                                    @click="openFolder(selectedDatasetInfo.images_dir!)"
                                    >{{ selectedDatasetInfo.images_count }} frames · {{ fmtBytes(selectedDatasetInfo.images_size!) }}</span
                                >
                                <span
                                    class="mi-desc"
                                    data-tip="Saved frame screenshots from collection. Click to open the folder in Finder."
                                    data-short="click to open folder"
                                ></span>
                            </div>
                        </template>
                    </template>
                </div>

                <!-- Collection settings -->
                <div class="settings-table mi-table">
                    <span class="sk">episodes</span>
                    <div class="sv mi-sv">
                        <span
                            class="clickable-field"
                            @click="openFieldEdit('collectEpisodes', $event)"
                            >{{ collectEpisodes }}</span
                        >
                        <span
                            class="mi-desc"
                            data-tip="Number of full episodes to collect. Each episode runs from game start until all lives are lost or max-steps is hit. More episodes = more diverse data but longer collection time."
                            data-short="episodes to collect"
                        ></span>
                    </div>

                    <span class="sk">max steps</span>
                    <div class="sv mi-sv">
                        <span
                            class="clickable-field"
                            @click="openFieldEdit('collectMaxSteps', $event)"
                            >{{ collectMaxSteps }}</span
                        >
                        <span
                            class="mi-desc"
                            data-tip="Maximum steps per episode. Each step applies one action for action_repeat frames (default 4 frames at 30fps ≈ 133ms). If the agent hasn't died by this limit, the episode is truncated. 200 steps ≈ 27 seconds of game time."
                            data-short="step limit per episode"
                        ></span>
                    </div>

                    <span class="sk">save frames</span>
                    <div class="sv mi-sv">
                        <label class="pipe-checkbox">
                            <input type="checkbox" v-model="saveFrames" />
                            <span class="pipe-checkbox-box"></span>
                        </label>
                        <span
                            class="mi-desc"
                            data-tip="Save screenshot PNGs alongside the transition data. Useful for debugging what the agent sees, but significantly increases disk usage."
                            data-short="persist frame images"
                        ></span>
                    </div>
                </div>

                <div v-if="collectError" class="tc-error pipe-err">
                    {{ collectError }}
                </div>

                <div class="pipe-nav">
                    <button @click="activeStep = prevStep(1)" class="btn-tiny">
                        ← back
                    </button>
                    <button
                        @click="activeStep = nextStep(1)"
                        class="btn-tiny pipe-next"
                        :disabled="!selectedDatasetInfo"
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
                        Train the JEPA world model encoder. It learns to
                        predict future latent representations from current
                        observations and actions — no reward signal needed.
                        The frozen encoder then feeds into the DQN agent in
                        the next step.
                    </div>
                    <div
                        v-if="!training.isWorldTraining && training.worldJob?.status !== 'completed'"
                        class="pipe-panel-desc"
                        style="margin-top: 2px"
                    >
                        Phase 1 collects random browser data, phase 2 trains
                        the encoder on prediction loss. Start with 1000–2000
                        steps and increase if the loss is still decreasing.
                    </div>

                    <div class="settings-table">
                        <span class="sk">steps</span>
                        <div class="sv">
                            <span
                                class="clickable-field"
                                @click="openFieldEdit('worldSteps', $event)"
                                >{{ worldSteps }}</span
                            >
                            <div class="tc-field-desc">gradient updates for the world model</div>
                        </div>

                        <span class="sk">batch</span>
                        <div class="sv">
                            <span
                                class="clickable-field"
                                @click="openFieldEdit('worldBatch', $event)"
                            >
                                <span v-if="worldBatch">{{ worldBatch }}</span>
                                <span v-else class="cf-muted">auto</span>
                            </span>
                            <div class="tc-field-desc">sequences sampled per update</div>
                        </div>

                        <span class="sk">lr</span>
                        <div class="sv">
                            <span
                                class="clickable-field"
                                @click="openFieldEdit('worldLr', $event)"
                            >
                                <span v-if="worldLr">{{ worldLr }}</span>
                                <span v-else class="cf-muted">auto</span>
                            </span>
                            <div class="tc-field-desc">AdamW learning rate</div>
                        </div>

                        <span class="sk">pre-collect</span>
                        <div class="sv">
                            <span
                                class="clickable-field"
                                @click="
                                    openFieldEdit('worldCollectSteps', $event)
                                "
                            >
                                <span v-if="worldCollectSteps">{{
                                    worldCollectSteps
                                }}</span>
                                <span v-else class="cf-muted">auto</span>
                            </span>
                            <div class="tc-field-desc">random browser steps before training (opens game)</div>
                        </div>

                        <span class="sk">log every</span>
                        <div class="sv">
                            <span
                                class="clickable-field"
                                @click="openFieldEdit('worldDashEvery', $event)"
                                >{{ worldDashEvery }}</span
                            >
                            <div class="tc-field-desc">steps between metric snapshots</div>
                        </div>

                        <template
                            v-if="training.isWorldTraining && training.worldJob"
                        >
                            <span class="sk">run</span>
                            <span class="sv" style="color: var(--accent)">{{
                                training.worldJob.run_name
                            }}</span>

                            <template v-if="worldPhase === 'collecting'">
                                <span class="sk">collecting</span>
                                <span class="sv" style="font-family: 'IBM Plex Mono', monospace; font-size: 10px">
                                    {{ worldCollectDone }}/{{ worldCollectTotal }}
                                    <span style="color: var(--muted); margin-left: 4px">{{ worldEpisodes }} eps</span>
                                </span>
                            </template>
                            <template v-else-if="worldTrainStep > 0">
                                <span class="sk">loss</span>
                                <span class="sv" style="color: #4e89ba; font-family: 'IBM Plex Mono', monospace; font-size: 10px">
                                    {{ fmtNum(worldTrainLoss) }}
                                </span>
                                <span class="sk">step</span>
                                <span class="sv" style="font-family: 'IBM Plex Mono', monospace; font-size: 10px">
                                    {{ worldTrainStep }}/{{ training.worldJob.requested_steps }}
                                </span>
                            </template>
                        </template>

                        <template v-if="selectedDataset && !training.isWorldTraining">
                            <span class="sk">dataset</span>
                            <span class="sv" style="font-size: 10px; color: var(--accent)">{{ selectedDataset }}</span>
                        </template>
                        <div
                            v-if="worldError || training.worldJob?.error"
                            class="tc-error"
                            style="grid-column: 1/-1; font-size: 9px"
                        >
                            {{ worldError || training.worldJob?.error }}
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
                    Train or resume the selected run. Model, optimizer, replay,
                    and exploration settings come from the locked run snapshot.
                </div>

                <div class="settings-table">
                    <template v-if="isFrozenJepa">
                        <span class="sk">jepa ckpt</span>
                        <div class="sv">
                            <VDropdown
                                v-if="jepaCkptOptions.length > 0"
                                v-model="jepaCheckpoint"
                                :options="jepaCkptOptions"
                                full-width
                                compact
                            />
                            <span
                                v-else
                                class="clickable-field"
                                @click="openFieldEdit('jepaCheckpoint', $event)"
                            >
                                <span v-if="jepaCheckpoint">{{
                                    jepaCheckpoint
                                }}</span>
                                <span v-else class="cf-muted"
                                    >none — run train world first</span
                                >
                            </span>
                            <div class="tc-field-desc">world model checkpoint to freeze as encoder</div>
                        </div>
                    </template>

                    <template v-if="hasCheckpoint">
                        <span class="sk">from</span>
                        <VDropdown
                            v-model="selectedCheckpoint"
                            :options="checkpointOptions"
                            full-width
                            compact
                        />
                    </template>

                    <span class="sk">+ steps</span>
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
                                >→
                                {{ checkpointStep() + additionalSteps }}</span
                            >
                        </span>
                        <div class="tc-field-desc">environment steps for this training round</div>
                    </div>

                    <span class="sk">log every</span>
                    <div class="sv">
                        <span
                            class="clickable-field"
                            @click="openFieldEdit('logEvery', $event)"
                            >{{ logEvery }}</span
                            >
                        <div class="tc-field-desc">steps between metric log writes</div>
                    </div>

                    <span class="sk">headed</span
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
                    <button class="btn-tiny" @click="closeFieldEdit">
                        cancel
                    </button>
                    <button class="btn-accent-tiny" @click="saveFieldEdit">
                        save
                    </button>
                </div>
            </div>
        </template>
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

/* Data collection - checkbox */
.pipe-checkbox {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
}
.pipe-checkbox input {
    display: none;
}
.pipe-checkbox-box {
    width: 12px;
    height: 12px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--bg);
    position: relative;
    transition:
        background 0.15s,
        border-color 0.15s;
    flex-shrink: 0;
}
.pipe-checkbox input:checked + .pipe-checkbox-box {
    background: var(--accent);
    border-color: var(--accent);
}
.pipe-checkbox input:checked + .pipe-checkbox-box::after {
    content: "";
    position: absolute;
    left: 3px;
    top: 1px;
    width: 4px;
    height: 7px;
    border: solid var(--bg);
    border-width: 0 1.5px 1.5px 0;
    transform: rotate(45deg);
}
.pipe-checkbox-label {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* Inline dataset actions */
.pipe-ds-action {
    padding: 1px 6px !important;
    font-size: 9px !important;
}
.pipe-ds-delete {
    padding: 1px 6px;
    font-size: 9px;
    color: var(--red) !important;
    border-color: var(--red) !important;
    opacity: 0.6;
    transition: opacity 0.15s;
}
.pipe-ds-delete:hover {
    opacity: 1;
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

/* ========================================
   WIZARD
   ======================================== */

.wiz {
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: 3px;
    background: var(--bg);
    overflow: hidden;
}

/* Breadcrumb */
.wiz-crumb {
    display: flex;
    align-items: center;
    gap: 2px;
    padding: 0 10px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
}
.wiz-crumb-step {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 8px 10px;
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 600;
    color: var(--border);
    font-family: "Outfit", sans-serif;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: default;
    transition:
        color 0.15s,
        border-color 0.15s;
    margin-bottom: -1px;
}
.wiz-crumb-step.wcs-active {
    color: var(--accent);
    border-bottom-color: var(--accent);
}
.wiz-crumb-step.wcs-done {
    color: var(--green);
    cursor: pointer;
}
.wiz-crumb-step.wcs-done:hover {
    color: var(--text);
}
.wcs-num {
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    opacity: 0.8;
}

/* Step body */
.wiz-body {
    flex: 1;
    padding: 14px 14px 10px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    overflow-y: auto;
    max-height: 480px;
}
.wiz-body-settings {
    max-height: 540px;
}
.wiz-body-name {
    gap: 14px;
}

.wiz-step-header {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}
.wiz-step-title {
    font-family: "Outfit", sans-serif;
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.01em;
}
.wiz-step-subtitle {
    font-size: 9px;
    color: var(--muted);
    line-height: 1.4;
}

/* ── Algorithm list ── */
.wiz-algo-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.wac {
    border: 1px solid var(--border);
    border-left: 2px solid var(--cc, var(--border));
    border-radius: 2px;
    background: var(--surface);
    overflow: hidden;
    transition:
        border-color 0.12s,
        background 0.12s;
}
.wac.wac-selected {
    border-color: var(--cc);
    background: color-mix(in srgb, var(--cc) 6%, var(--bg));
}

.wac-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 10px 7px 8px;
    cursor: pointer;
    gap: 8px;
}
.wac-row:hover {
    background: var(--surface-2);
}
.wac.wac-selected .wac-row {
    background: transparent;
}
.wac-row-left {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
}
.wac-row-right {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-shrink: 0;
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
}

.wac-sel-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    flex-shrink: 0;
    transition: all 0.12s;
}
.wac.wac-selected .wac-sel-dot {
    background: var(--cc);
    border-color: var(--cc);
    box-shadow: 0 0 6px color-mix(in srgb, var(--cc) 50%, transparent);
}

.wac-name {
    font-family: "Outfit", sans-serif;
    font-size: 10px;
    font-weight: 700;
    color: var(--text);
    white-space: nowrap;
}
.wac-badge {
    font-family: "IBM Plex Mono", monospace;
    font-size: 7px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    padding: 1px 5px;
    border-radius: 2px;
    border: 1px solid;
    white-space: nowrap;
}
.wac-stat {
    color: var(--muted);
}
.wac-stat-best {
    color: var(--green);
    font-weight: 600;
}
.wac-stat-sep {
    color: var(--border);
    font-size: 8px;
}
.wac-stat-none {
    color: var(--border);
    font-style: italic;
}

.wac-toggle {
    background: none;
    border: none;
    color: var(--muted);
    font-size: 9px;
    cursor: pointer;
    padding: 1px 3px;
    line-height: 1;
    border-radius: 2px;
    transition:
        color 0.1s,
        background 0.1s;
}
.wac-toggle:hover {
    color: var(--text);
    background: var(--surface-2);
}

.wac-body {
    padding: 8px 10px 10px 20px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: color-mix(in srgb, var(--cc) 3%, var(--bg));
}
.wac-summary {
    font-size: 10px;
    color: var(--text);
    opacity: 0.85;
    line-height: 1.4;
    margin: 0;
}
.wac-bullets {
    margin: 0;
    padding-left: 14px;
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.wac-bullets li {
    font-size: 9px;
    color: var(--muted);
    line-height: 1.4;
}
.wac-meta {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding-top: 4px;
    border-top: 1px solid var(--border);
}
.wac-rec {
    font-size: 9px;
    color: var(--muted);
    font-style: italic;
}
.wac-history {
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--muted);
}
.wac-no-history {
    font-style: italic;
    color: var(--border);
}

/* ── LLM Advisor ── */
.wiz-advisor {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border: 1px dashed var(--border);
    border-radius: 2px;
    background: linear-gradient(
        135deg,
        rgba(196, 145, 82, 0.04),
        rgba(78, 137, 186, 0.04)
    );
    margin-top: 2px;
}
.wiz-advisor-icon {
    font-size: 12px;
    color: var(--accent);
    opacity: 0.55;
    flex-shrink: 0;
}
.wiz-advisor-text {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 1px;
}
.wiz-advisor-title {
    font-size: 10px;
    font-weight: 600;
    color: var(--text);
    font-family: "Outfit", sans-serif;
}
.wiz-advisor-sub {
    font-size: 9px;
    color: var(--muted);
}
.wiz-advisor-btn {
    font-size: 8px;
    font-family: "IBM Plex Mono", monospace;
    padding: 3px 8px;
    border: 1px solid var(--border);
    border-radius: 2px;
    background: transparent;
    color: var(--muted);
    cursor: not-allowed;
    opacity: 0.4;
    white-space: nowrap;
    flex-shrink: 0;
}

/* ── Settings accordion (step 1) ── */
.wss {
    border: 1px solid var(--border);
    border-radius: 2px;
    background: var(--surface);
    overflow: hidden;
}
.wss-head {
    display: flex;
    align-items: center;
    gap: 7px;
    width: 100%;
    padding: 6px 10px;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.1s;
}
.wss-head:hover {
    background: var(--surface-2);
}
.wss-arrow {
    font-size: 8px;
    color: var(--muted);
    width: 10px;
    flex-shrink: 0;
}
.wss-title {
    font-family: "Outfit", sans-serif;
    font-size: 9px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
    flex-shrink: 0;
}
.wss-hint {
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex: 1;
    min-width: 0;
}
.wss-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px 16px;
    padding: 8px 10px 10px;
    border-top: 1px solid var(--border);
}

/* Setting field */
.wsf {
    display: flex;
    flex-direction: column;
    gap: 3px;
    cursor: default;
}
.wsf-wide {
    grid-column: 1 / -1;
}
.wsf-l {
    font-size: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    font-weight: 600;
    font-family: "Outfit", sans-serif;
}
.wsf-i {
    background: transparent;
    border: none;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 0;
    outline: none;
    width: 100%;
    transition: border-color 0.15s;
    -moz-appearance: textfield;
}
.wsf-i:focus {
    border-bottom-color: var(--accent);
}
.wsf-i::-webkit-outer-spin-button,
.wsf-i::-webkit-inner-spin-button {
    -webkit-appearance: none;
}
.wsf-sel {
    cursor: pointer;
}

.wsf-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
    padding-top: 2px;
}
.wsf-toggle input {
    display: none;
}
.wsf-toggle-track {
    width: 24px;
    height: 12px;
    background: var(--border);
    border-radius: 6px;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
}
.wsf-toggle input:checked + .wsf-toggle-track {
    background: var(--accent);
}
.wsf-toggle-thumb {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text);
    transition: transform 0.15s;
}
.wsf-toggle input:checked + .wsf-toggle-track .wsf-toggle-thumb {
    transform: translateX(12px);
}
.wsf-toggle-lbl {
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Name step ── */
.wiz-name-preview {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 10px;
}
.wnp-sep {
    color: var(--border);
}
.wnp-game {
    color: var(--muted);
}

.wiz-name-input {
    width: 100%;
    background: transparent;
    border: none;
    border-bottom: 2px solid var(--border);
    color: var(--text);
    font-family: "IBM Plex Mono", monospace;
    font-size: 16px;
    font-weight: 500;
    padding: 8px 0 6px;
    outline: none;
    transition: border-color 0.2s;
    box-sizing: border-box;
}
.wiz-name-input:focus {
    border-bottom-color: var(--accent);
}
.wiz-name-input::placeholder {
    color: var(--border);
}

.wiz-name-hint {
    font-size: 9px;
    color: var(--muted);
    opacity: 0.65;
    display: flex;
    align-items: center;
    gap: 6px;
}
.wnh-key {
    font-family: "IBM Plex Mono", monospace;
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1px 4px;
    font-size: 8px;
}
.wnh-auto {
    cursor: pointer;
    color: var(--accent);
    opacity: 0.7;
    transition: opacity 0.15s;
}
.wnh-auto:hover {
    opacity: 1;
}

/* ── Wizard footer ── */
.wiz-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 12px 8px;
    border-top: 1px solid var(--border);
    background: var(--surface);
}
.wiz-cancel {
    background: none;
    border: none;
    color: var(--muted);
    font-size: 9px;
    cursor: pointer;
    padding: 3px 4px;
    font-family: "IBM Plex Mono", monospace;
    letter-spacing: 0.04em;
    transition: color 0.15s;
    opacity: 0.55;
}
.wiz-cancel:hover {
    color: var(--red);
    opacity: 1;
}
.wiz-nav {
    display: flex;
    align-items: center;
    gap: 5px;
}
.wiz-back {
    font-size: 9px;
    padding: 4px 9px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
    cursor: pointer;
    font-family: "IBM Plex Mono", monospace;
    transition: all 0.12s;
}
.wiz-back:hover {
    border-color: var(--muted);
    color: var(--text);
}
.wiz-next {
    font-size: 9px;
    padding: 4px 13px;
    background: rgba(196, 145, 82, 0.1);
    border: 1px solid rgba(196, 145, 82, 0.25);
    border-radius: 2px;
    color: var(--accent);
    cursor: pointer;
    font-family: "IBM Plex Mono", monospace;
    font-weight: 600;
    letter-spacing: 0.04em;
    transition: all 0.12s;
}
.wiz-next:hover {
    background: rgba(196, 145, 82, 0.18);
    border-color: var(--accent);
}
.wiz-create {
    font-size: 9px;
    padding: 5px 13px;
    background: var(--accent);
    border: none;
    border-radius: 2px;
    color: var(--bg);
    cursor: pointer;
    font-family: "IBM Plex Mono", monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    transition: background 0.12s;
}
.wiz-create:hover:not(:disabled) {
    background: #d4a870;
}
.wiz-create:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}
</style>
