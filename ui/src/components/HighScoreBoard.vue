<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue'

interface ScoreEntry {
  score: number
  player: string
  type?: string
  date?: string
  apm?: number
  algorithm?: string
  encoder?: string
  latent_dim?: string
  steps_trained?: string
}

const props = defineProps<{
  highscores: ScoreEntry[]
}>()

const emit = defineEmits<{ 'update:playerName': [name: string] }>()

const playerName = ref('')
const currentPage = ref(0)
const PAGE_SIZE = 7

const filterMode = ref<'all' | 'human' | 'ai'>('all')
const filterAlgo = ref<string>('all')

onMounted(() => {
  const stored = localStorage.getItem('jeparl_player_name')
  if (stored) playerName.value = stored
  emit('update:playerName', playerName.value)
})

watch(playerName, name => {
  localStorage.setItem('jeparl_player_name', name)
  emit('update:playerName', name)
})

watch(() => props.highscores, () => {
  currentPage.value = 0
})

watch([filterMode, filterAlgo], () => {
  currentPage.value = 0
})

const uniqueAlgos = computed(() => {
  const algos = new Set<string>()
  for (const e of props.highscores) {
    if (isAi(e) && e.algorithm) algos.add(e.algorithm)
  }
  return Array.from(algos).sort()
})

const filteredScores = computed(() => {
  let scores = props.highscores
  if (filterMode.value === 'human') {
    scores = scores.filter(e => !isAi(e))
  } else if (filterMode.value === 'ai') {
    scores = scores.filter(e => isAi(e))
  }
  if (filterAlgo.value !== 'all') {
    scores = scores.filter(e => e.algorithm === filterAlgo.value)
  }
  return scores
})

const pageCount = computed(() => Math.max(1, Math.ceil(filteredScores.value.length / PAGE_SIZE)))
const pagedScores = computed(() => {
  const start = currentPage.value * PAGE_SIZE
  return filteredScores.value.slice(start, start + PAGE_SIZE)
})
const pageOffset = computed(() => currentPage.value * PAGE_SIZE)
const totalCount = computed(() => filteredScores.value.length)

function fmtDate(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  const mon = d.toLocaleString('default', { month: 'short' })
  const day = d.getDate()
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  return `${mon} ${day} ${h}:${m}`
}

function isAi(entry: ScoreEntry): boolean {
  return entry.type === 'ai' || entry.player === 'AI'
}

function parseAiPlayer(player: string): { run: string; checkpoint: string } | null {
  if (!player || player === 'AI') return null
  const sep = ' · '
  const idx = player.indexOf(sep)
  if (idx === -1) return { run: player, checkpoint: '' }
  return { run: player.slice(0, idx), checkpoint: player.slice(idx + sep.length) }
}

function fmtAlgo(algo: string | undefined): string {
  if (!algo) return ''
  if (algo === 'frozen_jepa_dqn') return 'JEPA+DQN'
  if (algo === 'dqn') return 'DQN'
  if (algo === 'linear_q') return 'LinQ'
  return algo.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function fmtSteps(steps: string | undefined): string {
  if (!steps) return ''
  const n = Number(steps)
  if (isNaN(n)) return steps
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function podiumClass(globalIndex: number): string {
  if (globalIndex === 0) return 'hs-podium-gold'
  if (globalIndex === 1) return 'hs-podium-silver'
  if (globalIndex === 2) return 'hs-podium-bronze'
  return ''
}

function rankStyle(globalIndex: number): string {
  if (globalIndex === 0) return 'hs-rank hs-rank-gold'
  if (globalIndex === 1) return 'hs-rank hs-rank-silver'
  if (globalIndex === 2) return 'hs-rank hs-rank-bronze'
  return 'hs-rank'
}
</script>

<template>
  <section class="hs-board">
    <div class="hs-header">
      <div class="hs-header-left">
        <span class="hs-title">HIGH SCORES</span>
        <span v-if="totalCount" class="hs-count">{{ totalCount }}</span>
      </div>
      <input
        v-model="playerName"
        type="text"
        class="hs-name-input"
        placeholder="your name"
        maxlength="20"
        spellcheck="false"
      />
    </div>

    <!-- Filter bar -->
    <div class="hs-filters">
      <div class="hs-filter-group">
        <button
          class="hs-filter-btn"
          :class="{ active: filterMode === 'all' }"
          @click="filterMode = 'all'"
        >ALL</button>
        <button
          class="hs-filter-btn"
          :class="{ active: filterMode === 'human' }"
          @click="filterMode = 'human'"
        >HUMAN</button>
        <button
          class="hs-filter-btn"
          :class="{ active: filterMode === 'ai' }"
          @click="filterMode = 'ai'"
        >AI</button>
      </div>
      <div v-if="uniqueAlgos.length > 0" class="hs-filter-group">
        <select v-model="filterAlgo" class="hs-algo-select">
          <option value="all">All algos</option>
          <option v-for="algo in uniqueAlgos" :key="algo" :value="algo">
            {{ fmtAlgo(algo) }}
          </option>
        </select>
      </div>
    </div>

    <div v-if="highscores.length === 0" class="hs-empty">
      <div class="hs-empty-dash">&mdash;&mdash;&mdash;</div>
      <div class="hs-empty-text">waiting for scores</div>
    </div>

    <div v-else-if="filteredScores.length === 0" class="hs-empty">
      <div class="hs-empty-text">no scores match filter</div>
    </div>

    <div v-else class="hs-table">
      <div class="hs-row hs-row-head">
        <span class="hs-col-rank hs-col-label">#</span>
        <span class="hs-col-player hs-col-label">PLAYER</span>
        <span class="hs-col-algo hs-col-label">ALGO</span>
        <span class="hs-col-meta hs-col-label">STEPS</span>
        <span class="hs-col-meta hs-col-label">APM</span>
        <span class="hs-col-date hs-col-label">DATE</span>
        <span class="hs-col-score hs-col-label">SCORE</span>
      </div>
      <div
        v-for="(entry, idx) in pagedScores"
        :key="pageOffset + idx"
        class="hs-row"
        :class="podiumClass(pageOffset + idx)"
      >
        <span :class="rankStyle(pageOffset + idx)">{{ pageOffset + idx + 1 }}</span>

        <span v-if="isAi(entry)" class="hs-col-player hs-player-ai">
          <span class="hs-ai-badge">AI</span>
          <span class="hs-ai-name">{{ parseAiPlayer(entry.player)?.checkpoint || parseAiPlayer(entry.player)?.run || entry.player }}</span>
          <span v-if="parseAiPlayer(entry.player)?.run && parseAiPlayer(entry.player)?.checkpoint" class="hs-ai-run-sub">{{ parseAiPlayer(entry.player)!.run }}</span>
        </span>
        <span v-else class="hs-col-player hs-player-human">{{ entry.player }}</span>

        <span class="hs-col-algo">{{ isAi(entry) ? fmtAlgo(entry.algorithm) : '' }}</span>

        <span class="hs-col-meta">{{ fmtSteps(entry.steps_trained) || '—' }}</span>

        <span class="hs-col-meta">{{ entry.apm ?? '—' }}</span>

        <span class="hs-col-date">{{ fmtDate(entry.date) }}</span>

        <span class="hs-col-score">{{ entry.score }}</span>
      </div>
    </div>

    <div v-if="pageCount > 1" class="hs-pagination">
      <button class="hs-page-btn" :disabled="currentPage === 0" @click="currentPage--">&lsaquo;</button>
      <div class="hs-page-dots">
        <span
          v-for="p in pageCount"
          :key="p"
          class="hs-page-dot"
          :class="{ active: p - 1 === currentPage }"
          @click="currentPage = p - 1"
        />
      </div>
      <button class="hs-page-btn" :disabled="currentPage >= pageCount - 1" @click="currentPage++">&rsaquo;</button>
    </div>
  </section>
</template>

<style scoped>
.hs-board {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0 0 2px 2px;
}

/* ── Header ── */
.hs-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 7px 10px;
  border-bottom: 1px solid var(--border);
}
.hs-header-left {
  display: flex;
  align-items: center;
  gap: 6px;
}
.hs-title {
  font-size: 9px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
}
.hs-count {
  font-family: "IBM Plex Mono", monospace;
  font-size: 8px;
  color: var(--accent);
  background: rgba(196, 145, 82, 0.1);
  border-radius: 2px;
  padding: 0 4px;
  font-weight: 600;
}
.hs-name-input {
  width: 90px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 2px;
  color: #27d6a0;
  padding: 2px 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  outline: none;
  text-align: right;
  transition: border-color 0.15s;
}
.hs-name-input:focus {
  border-color: var(--accent);
}
.hs-name-input::placeholder {
  color: var(--muted);
  opacity: 0.4;
}

/* ── Filters ── */
.hs-filters {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px;
  border-bottom: 1px solid var(--border);
  background: rgba(0, 0, 0, 0.15);
}
.hs-filter-group {
  display: flex;
  align-items: center;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.hs-filter-btn {
  background: transparent;
  border: none;
  color: var(--muted);
  padding: 3px 8px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 8px;
  font-weight: 600;
  letter-spacing: 0.06em;
  cursor: pointer;
  transition: all 0.15s;
  border-right: 1px solid var(--border);
}
.hs-filter-btn:last-child {
  border-right: none;
}
.hs-filter-btn:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.03);
}
.hs-filter-btn.active {
  color: var(--accent);
  background: rgba(196, 145, 82, 0.1);
}
.hs-algo-select {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  color: var(--text);
  padding: 2px 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 9px;
  outline: none;
  cursor: pointer;
}
.hs-algo-select:focus {
  border-color: var(--accent);
}

/* ── Empty ── */
.hs-empty {
  padding: 14px 8px 12px;
  text-align: center;
}
.hs-empty-dash {
  font-family: "IBM Plex Mono", monospace;
  font-size: 14px;
  letter-spacing: 0.15em;
  color: var(--border);
  margin-bottom: 4px;
}
.hs-empty-text {
  font-size: 8px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  opacity: 0.6;
}

/* ── Table grid ── */
.hs-table {
  display: grid;
}
.hs-row {
  /* rank fixed, player flex, algo fixed, steps+apm fixed, date flex, score fixed */
  display: grid;
  grid-template-columns: 20px 1fr 56px 38px 30px 1fr 48px;
  gap: 0 6px;
  padding: 5px 10px;
  font-size: 11px;
  border-left: 3px solid transparent;
  align-items: center;
  position: relative;
  transition: background 0.2s;
}
.hs-row:not(.hs-row-head):hover {
  background: rgba(255, 255, 255, 0.015);
}
.hs-row-head {
  padding: 4px 10px 3px;
  border-bottom: 1px solid var(--border);
  border-left: none;
}
.hs-col-label {
  font-size: 7px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  opacity: 0.5;
}

/* ── Shared column overflow ── */
.hs-col-rank,
.hs-col-player,
.hs-col-algo,
.hs-col-meta,
.hs-col-date,
.hs-col-score {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ── Rank ── */
.hs-col-rank,
.hs-rank {
  font-family: "IBM Plex Mono", monospace;
  font-size: 9px;
  text-align: center;
  color: var(--muted);
  line-height: 1;
}
.hs-rank-gold {
  color: #ffd166;
  font-weight: 700;
  text-shadow: 0 0 8px rgba(255, 209, 102, 0.35);
  font-size: 10px;
}
.hs-rank-silver {
  color: #c0c8d8;
  font-weight: 600;
  text-shadow: 0 0 5px rgba(192, 200, 216, 0.2);
}
.hs-rank-bronze {
  color: #cd7f32;
  font-weight: 600;
  text-shadow: 0 0 5px rgba(205, 127, 50, 0.2);
}

/* ── Player ── */
.hs-col-player {
  min-width: 0;
}
.hs-player-ai {
  display: flex;
  align-items: baseline;
  gap: 3px;
  line-height: 1.3;
}
.hs-ai-badge {
  font-family: "IBM Plex Mono", monospace;
  font-size: 6px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #64a8ff;
  background: rgba(100, 168, 255, 0.08);
  border: 1px solid rgba(100, 168, 255, 0.2);
  border-radius: 2px;
  padding: 0 3px;
  line-height: 1.5;
  flex-shrink: 0;
}
.hs-ai-name {
  font-family: "IBM Plex Mono", monospace;
  font-size: 9px;
  font-weight: 600;
  color: #64a8ff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}
.hs-ai-run-sub {
  display: none;
}
.hs-player-human {
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  font-weight: 600;
  color: #27d6a0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1;
}

/* ── Algo ── */
.hs-col-algo {
  font-family: "IBM Plex Mono", monospace;
  font-size: 7px;
  font-weight: 600;
  color: #64a8ff;
  text-transform: uppercase;
  letter-spacing: 0.02em;
}

/* ── Steps / APM (shared) ── */
.hs-col-meta {
  font-family: "IBM Plex Mono", monospace;
  font-size: 8px;
  color: var(--muted);
  text-align: right;
  font-variant-numeric: tabular-nums;
}

/* ── Date ── */
.hs-col-date {
  font-family: "IBM Plex Mono", monospace;
  font-size: 8px;
  color: var(--muted);
  opacity: 0.7;
}

/* ── Score ── */
.hs-col-score {
  font-family: "IBM Plex Mono", monospace;
  font-size: 11px;
  color: var(--text);
  text-align: right;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}

/* ── Podium: gradient fading into noise ── */
.hs-podium-gold,
.hs-podium-silver,
.hs-podium-bronze {
  position: relative;
  overflow: hidden;
}
.hs-podium-gold::before,
.hs-podium-silver::before,
.hs-podium-bronze::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}
.hs-podium-gold > *,
.hs-podium-silver > *,
.hs-podium-bronze > * {
  position: relative;
  z-index: 1;
}

.hs-podium-gold::before {
  background:
    linear-gradient(135deg,
      rgba(255, 209, 102, 0.09) 0%,
      rgba(255, 209, 102, 0.04) 35%,
      transparent 70%
    );
  border-left-color: #ffd166;
}
.hs-podium-gold::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.25;
  background: url("data:image/svg+xml,%3Csvg viewBox='0 0 128 128' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
  mask-image: linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 35%, transparent 65%);
  -webkit-mask-image: linear-gradient(135deg, rgba(0,0,0,0.7) 0%, rgba(0,0,0,0.3) 35%, transparent 65%);
}
.hs-podium-gold {
  border-left-color: #ffd166;
}
.hs-podium-gold .hs-col-score {
  font-weight: 700;
  color: #ffd166;
}
.hs-podium-gold .hs-ai-name {
  text-shadow: 0 0 8px rgba(100, 168, 255, 0.15);
}
.hs-podium-gold .hs-player-human {
  text-shadow: 0 0 8px rgba(39, 214, 160, 0.15);
}

.hs-podium-silver::before {
  background:
    linear-gradient(135deg,
      rgba(192, 200, 216, 0.06) 0%,
      rgba(192, 200, 216, 0.02) 35%,
      transparent 70%
    );
}
.hs-podium-silver::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.2;
  background: url("data:image/svg+xml,%3Csvg viewBox='0 0 128 128' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
  mask-image: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 35%, transparent 65%);
  -webkit-mask-image: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 35%, transparent 65%);
}
.hs-podium-silver {
  border-left-color: rgba(192, 200, 216, 0.3);
}
.hs-podium-silver .hs-col-score {
  font-weight: 700;
  color: #c0c8d8;
}

.hs-podium-bronze::before {
  background:
    linear-gradient(135deg,
      rgba(205, 127, 50, 0.06) 0%,
      rgba(205, 127, 50, 0.02) 35%,
      transparent 70%
    );
}
.hs-podium-bronze::after {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  opacity: 0.18;
  background: url("data:image/svg+xml,%3Csvg viewBox='0 0 128 128' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.06'/%3E%3C/svg%3E");
  mask-image: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 35%, transparent 65%);
  -webkit-mask-image: linear-gradient(135deg, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 35%, transparent 65%);
}
.hs-podium-bronze {
  border-left-color: rgba(205, 127, 50, 0.3);
}
.hs-podium-bronze .hs-col-score {
  font-weight: 700;
  color: #cd7f32;
}

/* ── Pagination ── */
.hs-pagination {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 5px 8px;
  border-top: 1px solid var(--border);
}
.hs-page-btn {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--muted);
  padding: 1px 6px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 10px;
  cursor: pointer;
  transition: all 0.15s;
  line-height: 1;
}
.hs-page-btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent);
}
.hs-page-btn:disabled {
  opacity: 0.2;
  cursor: default;
}
.hs-page-dots {
  display: flex;
  align-items: center;
  gap: 3px;
}
.hs-page-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--border);
  cursor: pointer;
  transition: all 0.15s;
}
.hs-page-dot.active {
  background: var(--accent);
  box-shadow: 0 0 4px rgba(196, 145, 82, 0.4);
}
.hs-page-dot:hover:not(.active) {
  background: var(--muted);
}
</style>
