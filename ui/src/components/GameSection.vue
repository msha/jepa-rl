<script setup lang="ts">
import {
    ref,
    computed,
    watch,
    onBeforeUnmount,
    onMounted,
    nextTick,
} from "vue";
import { usePolling } from "../composables/usePolling";
import { useConfigStore } from "../stores/config";
import VDropdown from "./VDropdown.vue";
import type { Job, EvalJob, CollectJob } from "../stores/training";

const configStore = useConfigStore();
const selectedConfig = ref("");

const configOptions = computed(() =>
    configStore.configs.map((c) => ({ value: c.path, label: c.name })),
);

onMounted(async () => {
    await configStore.loadConfigs();
    selectedConfig.value = configStore.currentPath;
});

async function onConfigChange() {
    if (!selectedConfig.value) return;
    try {
        await configStore.switchConfig(selectedConfig.value);
        reloadGame();
    } catch (e) {
        console.error(e);
    }
}

const props = defineProps<{
    job: Job | null;
    evalJob: EvalJob | null;
    collectJob: CollectJob | null;
    resetKey: string;
    actionKeys: string[];
    steps: Record<string, unknown>[];
    playerName: string;
    modelInfo: Record<string, unknown>;
}>();

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
const isCollecting = computed(
    () =>
        !isTraining.value &&
        !isEvaluating.value &&
        (!!props.collectJob?.running ||
            props.collectJob?.status === "running" ||
            props.collectJob?.status === "starting"),
);

const showGame = ref(true);
const gameStats = ref("");
const gameFrame = ref<HTMLIFrameElement | null>(null);
const gameSectionEl = ref<HTMLDivElement | null>(null);
const gameReloadToken = ref(0);
const evalClientError = ref("");
const gameHighscores = ref<{ score: number; player: string }[]>([]);
const evalQValues = ref<number[]>([]);
const heatmapCanvas = ref<HTMLCanvasElement | null>(null);

const emit = defineEmits<{
    highscores: [scores: { score: number; player: string }[]];
}>();

function onGameMessage(event: MessageEvent) {
    if (event.data?.type !== "jeparl-scores") return;
    gameHighscores.value = event.data.scores || [];
    emit("highscores", gameHighscores.value);
}
onMounted(() => {
    window.addEventListener("message", onGameMessage);
    document.addEventListener("click", onDocumentClick, true);
});
onBeforeUnmount(() => {
    window.removeEventListener("message", onGameMessage);
    document.removeEventListener("click", onDocumentClick, true);
});

function onDocumentClick(e: MouseEvent) {
    const el = gameSectionEl.value;
    if (!el) return;
    if (!el.contains(e.target as Node)) {
        gameFocused.value = false;
    }
}

let aiInterval: ReturnType<typeof window.setInterval> | null = null;
let aiStepInFlight = false;
let collectInterval: ReturnType<typeof window.setInterval> | null = null;
let collectStepInFlight = false;
let lastMirroredTrainingStep = 0;
let trainingPlaybackTimers: ReturnType<typeof window.setTimeout>[] = [];

const gameTitle = computed(() => {
    if (isTraining.value) return "training";
    if (isEvaluating.value) return "AI playing";
    if (isCollecting.value) return "collecting";
    if (props.evalJob?.result) return "eval done";
    return "game view";
});

const currentTrainingStep = computed(() => {
    const last = props.steps[props.steps.length - 1];
    return typeof last?.step === "number" ? last.step : 0;
});

const gameStatus = computed(() => {
    if (evalClientError.value) return `eval error: ${evalClientError.value}`;
    if (isTraining.value)
        return `training · step ${currentTrainingStep.value}/${props.job?.requested_steps ?? 0}`;
    if (isEvaluating.value) {
        const episode = (props.evalJob?.episode_count ?? 0) + 1;
        const target = props.evalJob?.episodes_target ?? 0;
        const suffix = target ? ` · ep ${episode}/${target}` : "";
        return `AI playing live${suffix}`;
    }
    if (isCollecting.value) {
        const done = props.collectJob?.episodes_done ?? 0;
        const target = props.collectJob?.episodes_target ?? 0;
        return `collecting · ep ${done}/${target}`;
    }
    if (props.job?.status === "completed" || props.job?.status === "stopped")
        return `${props.job.status}`;
    if (props.job?.status === "error") return `error: ${props.job.error || ""}`;
    if (props.evalJob?.status === "error")
        return `eval error: ${props.evalJob.error || ""}`;
    if (props.evalJob?.result) {
        const r = props.evalJob.result;
        return `eval · mean ${fmtNum(r.mean_score)} · best ${fmtNum(r.best_score)}`;
    }
    return "manual play";
});

const hasArrow = (dir: string) => props.actionKeys.some((k) => k.includes(dir));
const showUp = computed(() => hasArrow("ArrowUp"));
const showDown = computed(() => hasArrow("ArrowDown"));
const showLeft = computed(() => hasArrow("ArrowLeft"));
const showRight = computed(() => hasArrow("ArrowRight"));

const isIdle = computed(
    () => !isTraining.value && !isEvaluating.value && !isCollecting.value,
);

const heldDirs = ref<string[]>([]);
let clickLitTimer: ReturnType<typeof window.setTimeout> | null = null;
const gameFocused = ref(false);
const inputMode = ref<"arrows" | "wasd">("arrows");
const recentlyActive = ref(false);
let activeTimer: ReturnType<typeof window.setTimeout> | null = null;
const spaceNeeded = ref(true);
const prevLives = ref(3);

const litDir = computed(() => {
    const list = heldDirs.value;
    return list.length > 0 ? list[list.length - 1] : null;
});

function pressDir(dir: string) {
    if (!heldDirs.value.includes(dir)) {
        heldDirs.value = [...heldDirs.value, dir];
    }
}

function releaseDir(dir: string) {
    heldDirs.value = heldDirs.value.filter((d) => d !== dir);
}

// For button clicks — auto-release after 200ms
function lightDir(dir: string) {
    pressDir(dir);
    if (clickLitTimer) clearTimeout(clickLitTimer);
    clickLitTimer = setTimeout(() => releaseDir(dir), 200);
}

// Space is a continuous gameplay action (e.g. shoot in asteroids) vs start-only (breakout serve)
const spaceIsContinuous = computed(() =>
    props.actionKeys.some((k) => k.includes("Space") && k.includes("+")),
);
const spaceActive = computed(
    () => spaceNeeded.value || spaceIsContinuous.value,
);
const spaceButtonClass = computed(() => ({
    "ctrl-flicker": spaceNeeded.value && !gameFocused.value,
    "ctrl-act-secondary": !spaceActive.value,
}));

const gameStarted = ref(false);
const gameOver = ref(false);
const currentScore = ref(0);

function markActive() {
    recentlyActive.value = true;
    if (activeTimer) clearTimeout(activeTimer);
    activeTimer = setTimeout(() => {
        recentlyActive.value = false;
    }, 2000);
}

const keyModeClass = computed(() => {
    if (!isIdle.value || gameStarted.value) {
        return inputMode.value === "wasd" ? "mode-wasd" : "mode-arrows";
    }
    return "idle-morph";
});

function mapKey(e: KeyboardEvent): { key: string; code: string } | null {
    const k = e.key;
    if (k === "w" || k === "W") return { key: "ArrowUp", code: "ArrowUp" };
    if (k === "a" || k === "A") return { key: "ArrowLeft", code: "ArrowLeft" };
    if (k === "s" || k === "S") return { key: "ArrowDown", code: "ArrowDown" };
    if (k === "d" || k === "D")
        return { key: "ArrowRight", code: "ArrowRight" };
    if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(k))
        return { key: k, code: k };
    if (k === " ") return { key: " ", code: "Space" };
    if (k === "r" || k === "R") return { key: "r", code: "KeyR" };
    return null;
}

function forwardKey(type: "keydown" | "keyup", key: string, code: string) {
    const win = gameFrame.value?.contentWindow;
    const doc = gameFrame.value?.contentDocument;
    if (!win || !doc) return;
    dispatchKey(win, doc, type, key, code);
}

function onKeyDown(e: KeyboardEvent) {
    if (e.key === "Escape") {
        gameFocused.value = false;
        return;
    }

    const k = e.key;
    if (["w", "W", "a", "A", "s", "S", "d", "D"].includes(k))
        inputMode.value = "wasd";
    else if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(k))
        inputMode.value = "arrows";

    const mapping = mapKey(e);
    if (!mapping) return;
    e.preventDefault();
    forwardKey("keydown", mapping.key, mapping.code);
    markActive();
    gameStarted.value = true;

    if (mapping.code === "ArrowUp") pressDir("up");
    else if (mapping.code === "ArrowDown") pressDir("down");
    else if (mapping.code === "ArrowLeft") pressDir("left");
    else if (mapping.code === "ArrowRight") pressDir("right");
    else if (mapping.code === "Space") spaceNeeded.value = false;
}

function onKeyUp(e: KeyboardEvent) {
    const mapping = mapKey(e);
    if (!mapping) return;
    e.preventDefault();
    forwardKey("keyup", mapping.key, mapping.code);

    if (mapping.code === "ArrowUp") releaseDir("up");
    else if (mapping.code === "ArrowDown") releaseDir("down");
    else if (mapping.code === "ArrowLeft") releaseDir("left");
    else if (mapping.code === "ArrowRight") releaseDir("right");
}

function focusGameSection(e?: MouseEvent) {
    if (e) {
        const target = e.target as HTMLElement;
        if (
            target.tagName === "SELECT" ||
            target.tagName === "INPUT" ||
            target.closest(".vdd")
        )
            return;
    }
    gameFocused.value = true;
    nextTick(() => gameSectionEl.value?.focus());
}

const sectionClass = computed(() => ({
    "game-section": true,
    "game-focused": gameFocused.value || !isIdle.value,
    training: isTraining.value,
    evaluating: isEvaluating.value,
    collecting: isCollecting.value,
}));

const trainFrameSrc = ref("");
const gameSrc = computed(() => {
    const ts = gameReloadToken.value ? `&ts=${gameReloadToken.value}` : "";
    const algo = encodeURIComponent(String(props.modelInfo.algorithm || ""));
    const encType = encodeURIComponent(
        String(props.modelInfo.encoder_type || ""),
    );
    const latentDim = props.modelInfo.latent_dim ?? "";
    if (isTraining.value) {
        const runName = encodeURIComponent(props.job?.run_name || "");
        const stepsTrained =
            props.steps.length > 0
                ? props.steps[props.steps.length - 1].step
                : 0;
        return `/game?embed&run_name=${runName}&algorithm=${algo}&encoder=${encType}&latent_dim=${latentDim}&steps_trained=${stepsTrained}${ts}`;
    }
    if (isEvaluating.value) {
        const runName = encodeURIComponent(props.evalJob?.run_name || "");
        const ckpt = encodeURIComponent(props.evalJob?.checkpoint_name || "");
        return `/game?embed&run_name=${runName}&checkpoint=${ckpt}&algorithm=${algo}&encoder=${encType}&latent_dim=${latentDim}${ts}`;
    }
    const name = encodeURIComponent(props.playerName || "HUMAN");
    return `/game?embed&player_name=${name}${ts}`;
});

usePolling(async () => {
    // Poll frame
    if (isTraining.value) {
        trainFrameSrc.value = `/api/frame?ts=${Date.now()}`;
    }
}, 500);

usePolling(async () => {
    // Poll game stats from iframe
    try {
        const doc = gameFrame.value?.contentDocument;
        if (!doc) return;
        const score = doc.getElementById("score");
        const lives = doc.getElementById("lives");
        if (score && lives) {
            currentScore.value = parseInt(score.textContent || "0", 10);
            gameStats.value = `score <strong>${score.textContent || "0"}</strong> &nbsp; lives <strong>${lives.textContent || "3"}</strong>`;
            const newLives = parseInt(lives.textContent || "3", 10);
            if (newLives < prevLives.value) spaceNeeded.value = true;
            prevLives.value = newLives;
        }
        const doneEl = doc.querySelector("#status[data-state='done']");
        if (doneEl && gameStarted.value) {
            gameStarted.value = false;
            spaceNeeded.value = false;
            gameOver.value = true;
        }
    } catch {
        /* cross-origin */
    }
}, 500);

watch(
    isEvaluating,
    (active) => {
        if (active) {
            startAiLoop();
        } else {
            stopAiLoop(false);
        }
    },
    { immediate: true },
);

watch(
    isCollecting,
    (active) => {
        if (active) {
            startCollectLoop();
        } else {
            stopCollectLoop();
        }
    },
    { immediate: true },
);

watch(
    isTraining,
    (active) => {
        clearTrainingPlayback();
        if (active) {
            lastMirroredTrainingStep = 0;
            reloadGame();
            scheduleEpisodeStart();
        }
    },
    { immediate: true },
);

watch(() => props.steps, mirrorTrainingSteps, { deep: true });

onBeforeUnmount(() => {
    stopAiLoop(true);
    stopCollectLoop();
    clearTrainingPlayback();
    if (activeTimer) clearTimeout(activeTimer);
    if (clickLitTimer) clearTimeout(clickLitTimer);
});

function gameAction(action: string) {
    if (action === "space") {
        spaceNeeded.value = false;
        gameStarted.value = true;
    }
    const win = gameFrame.value?.contentWindow;
    const doc = gameFrame.value?.contentDocument;
    if (!win || !doc) return;
    const keyMap: Record<string, { key: string; code: string }> = {
        up: { key: "ArrowUp", code: "ArrowUp" },
        left: { key: "ArrowLeft", code: "ArrowLeft" },
        down: { key: "ArrowDown", code: "ArrowDown" },
        right: { key: "ArrowRight", code: "ArrowRight" },
        space: { key: " ", code: "Space" },
        reset: { key: "r", code: "KeyR" },
    };
    const mapping = keyMap[action];
    if (!mapping) return;
    dispatchKey(win, doc, "keydown", mapping.key, mapping.code);
    if (action !== "reset") {
        setTimeout(() => {
            dispatchKey(win, doc, "keyup", mapping.key, mapping.code);
        }, 120);
    }
}

function onResetClick() {
    gameAction("reset");
    reloadGame();
}

function stopAiMode() {
    if (isEvaluating.value) {
        stopAiLoop(true);
    } else if (isTraining.value) {
        fetch("/api/train/stop", { method: "POST" }).catch(() => {});
    } else if (isCollecting.value) {
        fetch("/api/collect-random/stop", { method: "POST" }).catch(() => {});
    }
}

function startAiLoop() {
    evalClientError.value = "";
    stopAiLoop(false);
    reloadGame();
    scheduleEpisodeStart();
    window.setTimeout(() => {
        if (!isEvaluating.value || aiInterval !== null) return;
        aiInterval = window.setInterval(runAiStep, 150);
    }, 750);
}

function stopAiLoop(notifyServer: boolean) {
    if (aiInterval !== null) {
        window.clearInterval(aiInterval);
        aiInterval = null;
    }
    aiStepInFlight = false;
    if (notifyServer) {
        fetch("/api/eval/stop", { method: "POST" }).catch(() => {});
    }
}

function startCollectLoop() {
    stopCollectLoop();
    reloadGame();
    scheduleCollectEpisodeStart();
    window.setTimeout(() => {
        if (!isCollecting.value || collectInterval !== null) return;
        collectInterval = window.setInterval(runCollectStep, 150);
    }, 750);
}

function stopCollectLoop() {
    if (collectInterval !== null) {
        window.clearInterval(collectInterval);
        collectInterval = null;
    }
    collectStepInFlight = false;
}

function scheduleCollectEpisodeStart(attempt = 0) {
    window.setTimeout(
        () => {
            if (!isCollecting.value) return;
            if (sendConfiguredKey(props.resetKey || "Space")) return;
            if (attempt < 10) scheduleCollectEpisodeStart(attempt + 1);
        },
        attempt === 0 ? 500 : 150,
    );
}

async function runCollectStep() {
    if (collectStepInFlight || !isCollecting.value) return;
    collectStepInFlight = true;
    try {
        const iframe = gameFrame.value;
        const doc = iframe?.contentDocument;
        const win = iframe?.contentWindow;
        if (!doc || !win) return;

        const doneEl = doc.querySelector("#status[data-state='done']");
        const isDone = !!doneEl;
        const scoreEl = doc.getElementById("score");
        const score = scoreEl
            ? Number.parseFloat(scoreEl.textContent || "0") || 0
            : 0;

        const res = await fetch("/api/collect-random/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ done: isDone, score }),
        });
        const data = (await res.json()) as {
            ok?: boolean;
            done?: boolean;
            action?: string;
            complete?: boolean;
            error?: string;
        };
        if (!res.ok || data.ok === false || data.error) {
            throw new Error(data.error || res.statusText);
        }
        if (data.complete) {
            stopCollectLoop();
            return;
        }
        const episodeDone = isDone || data.done;
        if (episodeDone) {
            window.setTimeout(() => {
                reloadGame();
                scheduleCollectEpisodeStart();
            }, 250);
            return;
        }
        if (data.action && data.action !== "noop") {
            applyKeyAction(win, doc, data.action);
        }
    } catch {
        stopCollectLoop();
    } finally {
        collectStepInFlight = false;
    }
}

function mirrorTrainingSteps() {
    if (!isTraining.value || !props.actionKeys.length) return;
    const unseen = props.steps
        .filter(
            (step) =>
                typeof step.step === "number" &&
                step.step > lastMirroredTrainingStep,
        )
        .sort((a, b) => Number(a.step) - Number(b.step));
    if (!unseen.length) return;

    const toReplay = unseen.length > 24 ? unseen.slice(-24) : unseen;
    lastMirroredTrainingStep = Number(unseen[unseen.length - 1].step);
    toReplay.forEach((step, index) => {
        const timer = window.setTimeout(
            () => mirrorTrainingStep(step),
            index * 70,
        );
        trainingPlaybackTimers.push(timer);
    });
}

function mirrorTrainingStep(step: Record<string, unknown>) {
    if (!isTraining.value) return;
    if (step.done) {
        reloadGame();
        scheduleEpisodeStart();
        return;
    }
    const actionIndex =
        typeof step.action === "number" ? step.action : Number(step.action);
    if (!Number.isFinite(actionIndex)) return;
    const action = props.actionKeys[actionIndex];
    if (!action || action === "noop") return;
    const win = gameFrame.value?.contentWindow;
    const doc = gameFrame.value?.contentDocument;
    if (!win || !doc) return;
    applyKeyAction(win, doc, action);
}

function clearTrainingPlayback() {
    trainingPlaybackTimers.forEach((timer) => window.clearTimeout(timer));
    trainingPlaybackTimers = [];
}

async function runAiStep() {
    if (aiStepInFlight || !isEvaluating.value) return;
    aiStepInFlight = true;
    try {
        const iframe = gameFrame.value;
        const doc = iframe?.contentDocument;
        const win = iframe?.contentWindow;
        if (!doc || !win) return;

        const doneEl = doc.querySelector("#status[data-state='done']");
        const isDone = !!doneEl;
        const scoreEl = doc.getElementById("score");
        const score = scoreEl
            ? Number.parseFloat(scoreEl.textContent || "0") || 0
            : 0;
        const canvas = doc.getElementById("game") as HTMLCanvasElement | null;
        if (!canvas) return;

        const frame = canvas.toDataURL("image/png").split(",")[1] || "";
        const res = await fetch("/api/eval/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame, done: isDone, score }),
        });
        const data = (await res.json()) as {
            ok?: boolean;
            action?: string;
            complete?: boolean;
            error?: string;
            q_values?: number[];
        };
        if (!res.ok || data.ok === false || data.error) {
            throw new Error(data.error || res.statusText);
        }
        if (data.q_values && data.q_values.length) {
            evalQValues.value = data.q_values;
        }
        if (data.complete) {
            stopAiLoop(false);
            return;
        }
        if (isDone) {
            window.setTimeout(() => {
                reloadGame();
                scheduleEpisodeStart();
            }, 250);
            return;
        }
        if (data.action && data.action !== "noop") {
            applyKeyAction(win, doc, data.action);
        }
    } catch (exc) {
        evalClientError.value =
            exc instanceof Error ? exc.message : String(exc);
        stopAiLoop(true);
    } finally {
        aiStepInFlight = false;
    }
}

function reloadGame() {
    gameReloadToken.value = Date.now();
    spaceNeeded.value = true;
    prevLives.value = 3;
    gameStarted.value = false;
    gameOver.value = false;
}

function scheduleEpisodeStart(attempt = 0) {
    window.setTimeout(
        () => {
            if (!isEvaluating.value && !isTraining.value) return;
            if (sendConfiguredKey(props.resetKey || "Space")) return;
            if (attempt < 10) scheduleEpisodeStart(attempt + 1);
        },
        attempt === 0 ? 500 : 150,
    );
}

function sendConfiguredKey(rawKey: string): boolean {
    const win = gameFrame.value?.contentWindow;
    const doc = gameFrame.value?.contentDocument;
    if (!win || !doc || !doc.getElementById("game")) return false;
    const { key, code } = normalizeKey(rawKey);
    dispatchKey(win, doc, "keydown", key, code);
    window.setTimeout(() => dispatchKey(win, doc, "keyup", key, code), 80);
    return true;
}

function applyKeyAction(win: Window, doc: Document, action: string) {
    const keyMap: Record<string, { key: string; code: string }> = {
        ArrowLeft: { key: "ArrowLeft", code: "ArrowLeft" },
        ArrowRight: { key: "ArrowRight", code: "ArrowRight" },
        Space: { key: " ", code: "Space" },
    };
    const keys = action
        .split("+")
        .map((raw) => keyMap[raw] ?? { key: raw, code: raw });
    keys.forEach(({ key, code }) =>
        dispatchKey(win, doc, "keydown", key, code),
    );
    window.setTimeout(() => {
        keys.slice()
            .reverse()
            .forEach(({ key, code }) =>
                dispatchKey(win, doc, "keyup", key, code),
            );
    }, 80);
}

function normalizeKey(rawKey: string): { key: string; code: string } {
    if (rawKey === "Space" || rawKey === " ")
        return { key: " ", code: "Space" };
    if (rawKey === "ArrowLeft") return { key: "ArrowLeft", code: "ArrowLeft" };
    if (rawKey === "ArrowRight")
        return { key: "ArrowRight", code: "ArrowRight" };
    if (rawKey.length === 1)
        return { key: rawKey, code: `Key${rawKey.toUpperCase()}` };
    return { key: rawKey, code: rawKey };
}

function dispatchKey(
    win: Window,
    doc: Document,
    type: "keydown" | "keyup",
    key: string,
    code: string,
) {
    const event = new KeyboardEvent(type, {
        key,
        code,
        bubbles: true,
        cancelable: true,
    });
    win.dispatchEvent(event);
    doc.dispatchEvent(
        new KeyboardEvent(type, { key, code, bubbles: true, cancelable: true }),
    );
}

function fmtNum(v: unknown): string {
    if (v == null || typeof v !== "number") return "—";
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    if (Math.abs(v) >= 10) return v.toFixed(2);
    return v.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

// Training stats
const trainLatestLoss = computed(() =>
    fmtNum(
        props.steps.length > 0
            ? props.steps[props.steps.length - 1].loss
            : null,
    ),
);
const trainLatestEpsilon = computed(() =>
    fmtNum(
        props.steps.length > 0
            ? props.steps[props.steps.length - 1].epsilon
            : null,
    ),
);
const trainLatestScore = computed(() =>
    fmtNum(
        props.steps.length > 0
            ? props.steps[props.steps.length - 1].score
            : null,
    ),
);
const trainLatestTdError = computed(() =>
    fmtNum(
        props.steps.length > 0
            ? props.steps[props.steps.length - 1].td_error
            : null,
    ),
);
const trainProgress = computed(() => {
    if (!props.job?.requested_steps) return 0;
    const step =
        props.steps.length > 0
            ? Number(props.steps[props.steps.length - 1].step)
            : 0;
    return Math.min(1, step / props.job.requested_steps);
});
const trainStepCount = computed(() =>
    props.steps.length > 0
        ? Number(props.steps[props.steps.length - 1].step)
        : 0,
);
const trainBestScore = computed(() => {
    let best = 0;
    for (const s of props.steps) {
        if (typeof s.score === "number" && s.score > best) best = s.score;
    }
    return best;
});

// Eval Q-value heatmap
function drawHeatmap() {
    const canvas = heatmapCanvas.value;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#08080a";
    ctx.fillRect(0, 0, w, h);

    const qVals = evalQValues.value;
    if (!qVals.length) return;

    // Generate a grid of "neurons" that activate based on Q-values
    const cols = Math.max(8, qVals.length * 3);
    const rows = 6;
    const cellW = w / cols;
    const cellH = h / rows;
    const cellPad = 1;

    // Seed a pseudo-random but deterministic pattern from Q-values
    const maxQ = Math.max(...qVals.map(Math.abs), 0.01);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            // Mix Q-values into activation intensity
            const qIdx = c % qVals.length;
            const qNorm = (qVals[qIdx] + maxQ) / (2 * maxQ);
            // Spatial variation using row/col modulation
            const spatial =
                0.3 + 0.7 * Math.sin((r * cols + c) * 0.7 + qNorm * Math.PI);
            // Activation intensity
            const activation = Math.max(
                0,
                Math.min(1, qNorm * 0.6 + spatial * 0.4),
            );
            const isSelected = qIdx === qVals.indexOf(Math.max(...qVals));

            let color: string;
            if (isSelected && activation > 0.5) {
                // Active neuron: accent gold
                const alpha = 0.3 + activation * 0.7;
                color = `rgba(196, 145, 82, ${alpha})`;
            } else if (activation > 0.6) {
                // Medium activation: blue
                const alpha = 0.2 + (activation - 0.6) * 2;
                color = `rgba(100, 168, 255, ${alpha})`;
            } else if (activation > 0.35) {
                // Low activation: dim blue
                color = `rgba(100, 168, 255, ${activation * 0.4})`;
            } else {
                // Dormant: very dim
                color = `rgba(100, 168, 255, ${activation * 0.15})`;
            }

            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(
                c * cellW + cellPad,
                r * cellH + cellPad,
                cellW - cellPad * 2,
                cellH - cellPad * 2,
                1,
            );
            ctx.fill();
        }
    }

    // Draw Q-value action labels at the bottom
    const labelH = 14;
    const barY = h - labelH;
    ctx.fillStyle = "#08080a";
    ctx.fillRect(0, barY, w, labelH);

    const barW = Math.min(60, (w - 20) / qVals.length);
    const totalBarW = qVals.length * (barW + 4);
    const startX = (w - totalBarW) / 2;

    qVals.forEach((q, i) => {
        const x = startX + i * (barW + 4);
        const qNorm = (q + maxQ) / (2 * maxQ);
        const barHeight = Math.max(2, qNorm * (labelH - 4));
        const isSelected = q === Math.max(...qVals);

        // Action label
        ctx.fillStyle = isSelected ? "#c49152" : "rgba(192, 200, 216, 0.4)";
        ctx.fillRect(x, barY + labelH - barHeight - 2, barW, barHeight);

        // Action key label
        if (props.actionKeys[i]) {
            ctx.fillStyle = isSelected
                ? "rgba(196, 145, 82, 0.9)"
                : "rgba(192, 200, 216, 0.3)";
            ctx.font = '7px "IBM Plex Mono", monospace';
            ctx.textAlign = "center";
            const label = props.actionKeys[i]
                .replace("Arrow", "")
                .replace("Space", "SPC")
                .substring(0, 4);
            ctx.fillText(label, x + barW / 2, barY + 8);
        }
    });
}

watch(evalQValues, () => {
    nextTick(drawHeatmap);
});

watch(isEvaluating, (active) => {
    if (!active) evalQValues.value = [];
});

// Redraw heatmap when canvas element appears or resizes
let heatmapObserver: ResizeObserver | null = null;
onMounted(() => {
    heatmapObserver = new ResizeObserver(() => {
        if (evalQValues.value.length) drawHeatmap();
    });
});
watch(heatmapCanvas, (el) => {
    if (heatmapObserver) {
        heatmapObserver.disconnect();
        if (el) heatmapObserver.observe(el);
    }
});
onBeforeUnmount(() => {
    heatmapObserver?.disconnect();
});
</script>

<template>
    <div
        :class="sectionClass"
        id="gameSection"
        ref="gameSectionEl"
        tabindex="-1"
        @click="focusGameSection"
        @keydown="onKeyDown"
        @keyup="onKeyUp"
    >
        <div class="game-header">
            <div class="gh-left">
                <span class="gh-title" id="gameTitle">{{ gameTitle }}</span>
                <span class="gh-status" id="gameStatus">{{ gameStatus }}</span>
            </div>
            <VDropdown
                v-model="selectedConfig"
                :options="configOptions"
                @change="onConfigChange"
                title="Available game configs in configs/games/"
                compact
            />
            <div class="gh-right">
                <span class="gh-stat" id="gameStats" v-html="gameStats"></span>
            </div>
        </div>

        <div class="game-body" v-show="showGame">
            <iframe
                ref="gameFrame"
                id="gameFrame"
                :src="gameSrc"
                scrolling="no"
                style="overflow: hidden"
                title="Game canvas"
            ></iframe>
            <div
                v-if="isIdle && !gameFocused && !gameOver"
                class="game-overlay game-overlay-click"
            >
                click to play
            </div>
            <div
                v-if="gameOver && isIdle"
                class="game-overlay game-over-overlay"
            >
                <span class="go-title">GAME OVER</span>
            </div>
            <div
                v-if="!isIdle"
                class="game-ai-badge"
                :class="{
                    'ai-training': isTraining,
                    'ai-eval': isEvaluating,
                    'ai-collect': isCollecting,
                }"
            >
                <span class="ai-badge-dot"></span>
                {{ gameTitle }}
            </div>
        </div>

        <img
            class="train-frame"
            id="trainFrame"
            :src="trainFrameSrc"
            alt="training view"
            title="Live training screenshot"
        />

        <!-- Contextual panel: always same height, content switches on mode -->
        <div class="game-panel" v-show="showGame">
            <!-- Training stats -->
            <div v-if="isTraining" class="panel-training">
                <div class="pt-metrics">
                    <div class="pt-metric">
                        <span class="pt-label">loss</span>
                        <span class="pt-val">{{ trainLatestLoss }}</span>
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">epsilon</span>
                        <span class="pt-val">{{ trainLatestEpsilon }}</span>
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">score</span>
                        <span class="pt-val">{{ trainLatestScore }}</span>
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">td err</span>
                        <span class="pt-val">{{ trainLatestTdError }}</span>
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">best</span>
                        <span class="pt-val pt-val-gold">{{
                            trainBestScore
                        }}</span>
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">step</span>
                        <span class="pt-val"
                            >{{ trainStepCount
                            }}<span class="pt-of"
                                >/{{ props.job?.requested_steps ?? 0 }}</span
                            ></span
                        >
                    </div>
                </div>
                <div class="pt-progress-track">
                    <div
                        class="pt-progress-fill"
                        :style="{ width: trainProgress * 100 + '%' }"
                    ></div>
                </div>
            </div>

            <!-- Eval heatmap -->
            <div v-else-if="isEvaluating" class="panel-eval">
                <canvas ref="heatmapCanvas" class="heatmap-canvas"></canvas>
            </div>

            <!-- Collecting: show collection stats -->
            <div v-else-if="isCollecting" class="panel-collecting">
                <div class="pt-metrics">
                    <div class="pt-metric">
                        <span class="pt-label">episodes</span>
                        <span class="pt-val"
                            >{{ props.collectJob?.episodes_done ?? 0
                            }}<span class="pt-of"
                                >/{{
                                    props.collectJob?.episodes_target ?? 0
                                }}</span
                            ></span
                        >
                    </div>
                    <div class="pt-metric">
                        <span class="pt-label">avg score</span>
                        <span class="pt-val">{{
                            fmtNum(props.collectJob?.mean_score)
                        }}</span>
                    </div>
                </div>
                <div class="pt-progress-track">
                    <div
                        class="pt-progress-fill"
                        :style="{
                            width:
                                ((props.collectJob?.episodes_done ?? 0) /
                                    (props.collectJob?.episodes_target || 1)) *
                                    100 +
                                '%',
                        }"
                    ></div>
                </div>
            </div>

            <!-- Normal game controls -->
            <div v-else class="game-controls" :class="keyModeClass">
                <button
                    @click="gameAction('space')"
                    class="ctrl-act"
                    :class="spaceButtonClass"
                    title="Space"
                >
                    space
                </button>
                <div class="ctrl-dpad-gap"></div>
                <div class="ctrl-dpad">
                    <div class="ctrl-dpad-row">
                        <button
                            @click="
                                gameAction('up');
                                lightDir('up');
                            "
                            class="ctrl-dir dir-up"
                            :class="{
                                'ctrl-dir-lit': litDir === 'up',
                                'ctrl-dir-unused': !showUp,
                            }"
                            title="ArrowUp"
                        >
                            <span class="lbl-arrow">&#9650;</span
                            ><span class="lbl-wasd">W</span>
                        </button>
                    </div>
                    <div class="ctrl-dpad-row">
                        <button
                            @click="
                                gameAction('left');
                                lightDir('left');
                            "
                            class="ctrl-dir dir-left"
                            :class="{
                                'ctrl-dir-lit': litDir === 'left',
                                'ctrl-dir-unused': !showLeft,
                            }"
                            title="ArrowLeft"
                        >
                            <span class="lbl-arrow">&#9664;</span
                            ><span class="lbl-wasd">A</span>
                        </button>
                        <button
                            @click="
                                gameAction('down');
                                lightDir('down');
                            "
                            class="ctrl-dir dir-down"
                            :class="{
                                'ctrl-dir-lit': litDir === 'down',
                                'ctrl-dir-unused': !showDown,
                            }"
                            title="ArrowDown"
                        >
                            <span class="lbl-arrow">&#9660;</span
                            ><span class="lbl-wasd">S</span>
                        </button>
                        <button
                            @click="
                                gameAction('right');
                                lightDir('right');
                            "
                            class="ctrl-dir dir-right"
                            :class="{
                                'ctrl-dir-lit': litDir === 'right',
                                'ctrl-dir-unused': !showRight,
                            }"
                            title="ArrowRight"
                        >
                            <span class="lbl-arrow">&#9654;</span
                            ><span class="lbl-wasd">D</span>
                        </button>
                    </div>
                </div>
                <div class="ctrl-spacer"></div>
                <button
                    @click="onResetClick"
                    class="ctrl-act"
                    :class="{ 'ctrl-act-secondary': !gameOver }"
                    title="R"
                >
                    reset
                </button>
            </div>

            <!-- Big STOP button for AI-controlled modes -->
            <button v-if="!isIdle" @click="stopAiMode" class="btn-stop-ai">
                STOP
            </button>
        </div>
    </div>
</template>

<style scoped>
.game-focused {
    animation: crt-flicker 0.06s infinite;
    background:
        radial-gradient(
            circle at center,
            transparent 25%,
            rgba(100, 168, 255, 0.06) 42%,
            rgba(100, 168, 255, 0.03) 52%,
            rgba(7, 9, 13, 0.4) 65%,
            rgba(7, 9, 13, 0.75) 80%,
            rgba(7, 9, 13, 0.95) 100%
        ),
        linear-gradient(
            180deg,
            rgba(100, 168, 255, 0.05) 0%,
            transparent 25%,
            transparent 75%,
            rgba(7, 9, 13, 0.5) 100%
        ),
        #08080a;
}
.game-focused .game-body {
    position: relative;
    z-index: 1;
    animation: crt-glow 1.4s ease-in-out infinite;
    outline: 1px solid rgba(100, 168, 255, 0.15);
    outline-offset: -1px;
    background:
        repeating-linear-gradient(
            0deg,
            rgba(0, 0, 0, 0.06) 0px,
            rgba(0, 0, 0, 0.06) 1px,
            transparent 1px,
            transparent 3px
        ),
        linear-gradient(
            180deg,
            rgba(100, 168, 255, 0.04) 0%,
            rgba(7, 9, 13, 0.95) 40%,
            #07090d 100%
        ),
        #07090d;
}
.game-body {
    position: relative;
}
@keyframes crt-flicker {
    0%,
    100% {
        filter: brightness(1);
    }
    30% {
        filter: brightness(0.985);
    }
    60% {
        filter: brightness(1.004);
    }
    80% {
        filter: brightness(0.99);
    }
}
@keyframes crt-glow {
    0%,
    100% {
        box-shadow:
            0 0 8px 3px rgba(100, 168, 255, 0.28),
            0 0 22px 8px rgba(100, 168, 255, 0.14),
            0 0 50px 18px rgba(100, 168, 255, 0.06),
            0 0 80px 28px rgba(100, 168, 255, 0.02);
    }
    50% {
        box-shadow:
            0 0 8px 3px rgba(100, 168, 255, 0.38),
            0 0 22px 8px rgba(100, 168, 255, 0.18),
            0 0 50px 18px rgba(100, 168, 255, 0.08),
            0 0 80px 28px rgba(100, 168, 255, 0.03);
    }
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
    font-family: "Outfit", sans-serif;
    cursor: pointer;
    transition: background 0.15s;
}
.game-overlay-click:hover {
    background: rgba(7, 9, 13, 0.25);
    color: rgba(244, 246, 251, 0.7);
}

/* AI mode badge */
.game-ai-badge {
    position: absolute;
    top: 6px;
    left: 6px;
    z-index: 3;
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 2px;
    font-family: "Outfit", sans-serif;
    font-size: 9px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    pointer-events: none;
    background: rgba(7, 9, 13, 0.7);
    border: 1px solid rgba(93, 158, 93, 0.3);
    color: #5d9e5d;
    backdrop-filter: blur(4px);
}
.ai-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #5d9e5d;
    animation: pulse 1.5s infinite;
}
.game-ai-badge.ai-training {
    border-color: rgba(93, 158, 93, 0.4);
    color: #5d9e5d;
}
.game-ai-badge.ai-training .ai-badge-dot {
    background: #5d9e5d;
}
.game-ai-badge.ai-eval {
    border-color: rgba(196, 145, 82, 0.4);
    color: var(--accent);
}
.game-ai-badge.ai-eval .ai-badge-dot {
    background: var(--accent);
}
.game-ai-badge.ai-collect {
    border-color: rgba(100, 168, 255, 0.4);
    color: #64a8ff;
}
.game-ai-badge.ai-collect .ai-badge-dot {
    background: #64a8ff;
}

.game-over-overlay {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 12%;
    background: transparent;
    pointer-events: none;
}
.go-title {
    font-family: "Outfit", sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #ff5a7a;
    text-shadow: 0 0 12px rgba(255, 90, 122, 0.4);
}
.game-focus-overlay {
    display: none;
}

/* Key label morphing */
.ctrl-dir {
    position: relative;
    overflow: hidden;
}
.lbl-arrow,
.lbl-wasd {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

/* Playing: instant opacity switch */
.mode-arrows .lbl-arrow {
    opacity: 1;
    transform: translate(-50%, -50%);
    transition: opacity 0.15s;
}
.mode-arrows .lbl-wasd {
    opacity: 0;
    transform: translate(-50%, -50%);
    transition: opacity 0.15s;
}
.mode-wasd .lbl-arrow {
    opacity: 0;
    transform: translate(-50%, -50%);
    transition: opacity 0.15s;
}
.mode-wasd .lbl-wasd {
    opacity: 1;
    transform: translate(-50%, -50%);
    transition: opacity 0.15s;
}

/* Idle: slide-left cascade (7s cycle = 3s arrows + 0.5s slide + 3s WASD + 0.5s slide)
   Cascade order: D(right)→S(down)→W(up)→A(left) with stagger delays */
@keyframes morph-out {
    0%,
    42.86% {
        transform: translate(-50%, -50%);
        opacity: 1;
    }
    50% {
        transform: translate(-50%, -50%) translateX(-120%);
        opacity: 0;
    }
    92.86% {
        transform: translate(-50%, -50%) translateX(120%);
        opacity: 0;
    }
    100% {
        transform: translate(-50%, -50%);
        opacity: 1;
    }
}
@keyframes morph-in {
    0%,
    42.86% {
        transform: translate(-50%, -50%) translateX(120%);
        opacity: 0;
    }
    50% {
        transform: translate(-50%, -50%);
        opacity: 1;
    }
    92.86% {
        transform: translate(-50%, -50%);
        opacity: 1;
    }
    100% {
        transform: translate(-50%, -50%) translateX(-120%);
        opacity: 0;
    }
}

.idle-morph .lbl-arrow {
    animation: morph-out 7s ease infinite;
}
.idle-morph .lbl-wasd {
    animation: morph-in 7s ease infinite;
}

/* Cascade: right=0ms, down=100ms, up=150ms, left=200ms */
.idle-morph .dir-down .lbl-arrow {
    animation-delay: 0.1s;
}
.idle-morph .dir-down .lbl-wasd {
    animation-delay: 0.1s;
}
.idle-morph .dir-up .lbl-arrow {
    animation-delay: 0.15s;
}
.idle-morph .dir-up .lbl-wasd {
    animation-delay: 0.15s;
}
.idle-morph .dir-left .lbl-arrow {
    animation-delay: 0.2s;
}
.idle-morph .dir-left .lbl-wasd {
    animation-delay: 0.2s;
}

.game-section:focus {
    outline: none;
}

/* Big STOP button for AI modes */
.btn-stop-ai {
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 78px;
    background: rgba(185, 82, 76, 0.12);
    border: none;
    border-left: 2px solid rgba(185, 82, 76, 0.5);
    color: var(--red);
    font-family: "Outfit", sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
    z-index: 10;
}
.btn-stop-ai:hover {
    background: rgba(185, 82, 76, 0.25);
    border-left-color: var(--red);
    color: #ff7a88;
    box-shadow: 0 0 12px rgba(185, 82, 76, 0.3);
}
.btn-stop-ai:active {
    background: rgba(185, 82, 76, 0.4);
    transform: scale(0.97);
}
.btn-stop-ai::after {
    content: "";
    position: absolute;
    left: -2px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--red);
    animation: stop-pulse 2s ease-in-out infinite;
}
@keyframes stop-pulse {
    0%,
    100% {
        opacity: 1;
    }
    50% {
        opacity: 0.3;
    }
}

.ctrl-dir-unused {
    opacity: 0.3 !important;
    cursor: default !important;
    pointer-events: none;
    color: var(--muted) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}
</style>
