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
const PAGE_SIZE = 10

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

const pageCount = computed(() => Math.ceil(props.highscores.length / PAGE_SIZE))
const pagedScores = computed(() => {
  const start = currentPage.value * PAGE_SIZE
  return props.highscores.slice(start, start + PAGE_SIZE)
})
const pageOffset = computed(() => currentPage.value * PAGE_SIZE)

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
      <span class="hs-title">HIGH SCORES</span>
      <div class="hs-name-field">
        <input
          v-model="playerName"
          type="text"
          class="hs-name-input"
          placeholder="your name"
          maxlength="20"
          spellcheck="false"
        />
      </div>
    </div>

    <div v-if="highscores.length === 0" class="hs-empty">
      <div class="hs-empty-dash">&mdash;&mdash;&mdash;</div>
      <div class="hs-empty-text">waiting for scores</div>
    </div>

    <div v-else class="hs-table">
      <div class="hs-row hs-row-head">
        <span class="hs-col-rank hs-col-label">#</span>
        <span class="hs-col-kind hs-col-label"></span>
        <span class="hs-col-player hs-col-label">NAME</span>
        <span class="hs-col-algo hs-col-label">ALGO</span>
        <span class="hs-col-steps hs-col-label">STEPS</span>
        <span class="hs-col-apm hs-col-label">APM</span>
        <span class="hs-col-date hs-col-label">DATE</span>
        <span class="hs-col-score hs-col-label">SCORE</span>
      </div>
      <div
        v-for="(entry, idx) in pagedScores"
        :key="pageOffset + idx"
        class="hs-row"
        :class="podiumClass(pageOffset + idx)"
      >
        <!-- Rank -->
        <span :class="rankStyle(pageOffset + idx)">{{ pageOffset + idx + 1 }}</span>

        <!-- Kind -->
        <span class="hs-col-kind" :class="isAi(entry) ? 'hs-kind-ai' : 'hs-kind-human'">
          {{ isAi(entry) ? 'AI' : '' }}
        </span>

        <!-- Player -->
        <span v-if="isAi(entry)" class="hs-col-player hs-player-ai">
          <span class="hs-ai-name">{{ parseAiPlayer(entry.player)?.checkpoint || parseAiPlayer(entry.player)?.run || entry.player }}</span>
          <span v-if="parseAiPlayer(entry.player)?.run && parseAiPlayer(entry.player)?.checkpoint" class="hs-ai-run-sub">{{ parseAiPlayer(entry.player)!.run }}</span>
        </span>
        <span v-else class="hs-col-player hs-player-human">{{ entry.player }}</span>

        <!-- Algorithm -->
        <span class="hs-col-algo">{{ isAi(entry) ? fmtAlgo(entry.algorithm) : '' }}</span>

        <!-- Steps trained -->
        <span class="hs-col-steps">{{ fmtSteps(entry.steps_trained) || '' }}</span>

        <!-- APM -->
        <span class="hs-col-apm">{{ entry.apm || '' }}</span>

        <!-- Date -->
        <span class="hs-col-date">{{ fmtDate(entry.date) }}</span>

        <!-- Score -->
        <span class="hs-col-score">{{ entry.score }}</span>
      </div>
    </div>

    <div v-if="pageCount > 1" class="hs-pagination">
      <button class="hs-page-btn" :disabled="currentPage === 0" @click="currentPage--">&lsaquo; prev</button>
      <span class="hs-page-info">{{ currentPage + 1 }} / {{ pageCount }}</span>
      <button class="hs-page-btn" :disabled="currentPage >= pageCount - 1" @click="currentPage++">next &rsaquo;</button>
    </div>
  </section>
</template>
