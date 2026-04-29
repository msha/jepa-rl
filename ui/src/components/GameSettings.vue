<script setup lang="ts">
import { computed, ref, onMounted, watch } from 'vue'

interface ScoreEntry {
  score: number
  player: string
  type?: string
  date?: string
}

const props = defineProps<{
  settings: [string, unknown][]
  highscores: ScoreEntry[]
}>()

const emit = defineEmits<{ 'update:playerName': [name: string] }>()

const playerName = ref('')

onMounted(() => {
  const stored = localStorage.getItem('jeparl_player_name')
  if (stored) playerName.value = stored
  emit('update:playerName', playerName.value)
})

watch(playerName, name => {
  localStorage.setItem('jeparl_player_name', name)
  emit('update:playerName', name)
})

const description = computed(() => {
  const entry = props.settings.find(([k]) => k === 'description')
  return entry ? String(entry[1]) : ''
})
const filteredSettings = computed(() => props.settings.filter(([k]) => k !== 'description'))

function fmtDate(iso: string | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  const mon = d.toLocaleString('default', { month: 'short' })
  const day = d.getDate()
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  return `${mon} ${day} ${h}:${m}`
}

function rankClass(i: number): string {
  if (i === 0) return 'hs-rank hs-gold'
  if (i === 1) return 'hs-rank hs-silver'
  if (i === 2) return 'hs-rank hs-bronze'
  return 'hs-rank'
}

function isAi(entry: ScoreEntry): boolean {
  return entry.type === 'ai' || entry.player === 'AI'
}
</script>

<template>
  <div class="settings-table" title="Game configuration settings for the active config">
    <template v-if="description">
      <span class="sk">description</span>
      <span class="sv game-desc">{{ description }}</span>
    </template>
    <template v-for="[key, val] in filteredSettings" :key="key">
      <span class="sk">{{ key }}</span>
      <span class="sv">{{ String(val) }}</span>
    </template>
  </div>
  <div class="highscore-board">
    <div class="hs-header-row">
      <div class="hs-title">HIGH SCORES</div>
      <div class="hs-name-field" title="Your name for human scores">
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
    <div v-if="highscores.length > 0">
      <div v-for="(entry, i) in highscores.slice(0, 5)" :key="i" class="hs-row" :class="{ 'hs-podium': i < 3 }">
        <span :class="rankClass(i)">{{ i + 1 }}</span>
        <span :class="['hs-player', isAi(entry) ? 'hs-ai' : 'hs-human']" :title="entry.player">{{ entry.player }}</span>
        <span class="hs-date">{{ fmtDate(entry.date) }}</span>
        <span class="hs-score">{{ entry.score }}</span>
      </div>
    </div>
    <div v-else class="hs-empty">no scores yet</div>
  </div>
</template>
