<script setup lang="ts">
import { computed } from 'vue'

interface CheckpointInfo {
  file: string
  label: string
}

const props = defineProps<{
  episodes: Record<string, unknown>[]
  summary: Record<string, unknown>
  checkpoints: CheckpointInfo[]
  hasCheckpoint: boolean
}>()

const recentEpisodes = computed(() => {
  return (props.episodes || []).slice(-30).reverse()
})

const allEpisodes = computed(() => props.episodes || [])

const personalBests = computed(() => {
  const set = new Set<number>()
  let runningBest = -Infinity
  for (const e of allEpisodes.value) {
    const s = (e.score as number) ?? 0
    if (s > runningBest) {
      runningBest = s
      set.add(e.episode as number)
    }
  }
  return set
})

const stats = computed(() => {
  const all = allEpisodes.value
  if (all.length === 0) return null
  const scores = all.map(e => (e.score as number) ?? 0)
  const returns = all.map(e => (e.return as number) ?? 0)
  const lengths = all.map(e => {
    const s = e.steps ?? (e.step as number)
    return typeof s === 'number' ? s : 0
  })
  const last5 = scores.slice(-5)
  const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length
  return {
    total: all.length,
    best: Math.max(...scores),
    meanScore: mean(scores),
    meanReturn: mean(returns),
    meanLength: mean(lengths),
    last5Avg: last5.length > 0 ? mean(last5) : 0,
    last10Avg: scores.slice(-10).length > 0 ? mean(scores.slice(-10)) : 0,
    lastScore: scores[scores.length - 1],
    trend: last5.length >= 3
      ? mean(last5) > mean(scores.slice(-10, -5) || last5) ? 'up' as const
        : mean(last5) < mean(scores.slice(-10, -5) || last5) ? 'down' as const : 'flat' as const
      : 'flat' as const,
  }
})

const bestScore = computed(() => {
  const all = allEpisodes.value
  if (all.length === 0) return -Infinity
  return Math.max(...all.map(e => (e.score as number) ?? 0))
})

const bestCheckpointEp = computed(() => {
  let bestIdx = -1
  let best = -Infinity
  for (let i = 0; i < allEpisodes.value.length; i++) {
    const s = (allEpisodes.value[i].score as number) ?? 0
    if (s > best) {
      best = s
      bestIdx = i
    }
  }
  return bestIdx >= 0 ? allEpisodes.value[bestIdx] : null
})

const bestCkptLabel = computed(() => {
  return props.checkpoints?.find(c => c.label === 'best')
})

function fmt(v: unknown): string {
  if (v == null || (typeof v === 'number' && Number.isNaN(v))) return '—'
  if (typeof v === 'number') {
    if (Math.abs(v) >= 1000) return v.toFixed(0)
    if (Math.abs(v) >= 10) return v.toFixed(1)
    return v.toFixed(2).replace(/0+$/, '').replace(/\.$/, '')
  }
  return String(v)
}

function rowClass(e: Record<string, unknown>): Record<string, boolean> {
  const score = (e.score as number) ?? 0
  const epNum = e.episode as number
  const isBest = score === bestScore.value
  const isPB = personalBests.value.has(epNum)
  const isHigh = stats.value && score >= stats.value.meanScore * 1.2
  const isLow = stats.value && score < stats.value.meanScore * 0.5
  return {
    'ep-best': isBest,
    'ep-pb': isPB && !isBest,
    'ep-high': !isBest && !isPB && !!isHigh,
    'ep-low': !isBest && !isPB && !!isLow,
  }
}

function scoreColor(score: number): string {
  if (score === bestScore.value) return '#ffd166'
  if (personalBests.value.size > 0 && stats.value && score >= stats.value.meanScore * 1.2) return 'var(--green, #5d9e5d)'
  if (stats.value && score < stats.value.meanScore * 0.5) return 'var(--red, #b9524c)'
  return 'var(--text)'
}

function trendIcon(t: 'up' | 'down' | 'flat'): string {
  if (t === 'up') return '↑'
  if (t === 'down') return '↓'
  return '→'
}

function trendColor(t: 'up' | 'down' | 'flat'): string {
  if (t === 'up') return 'var(--green, #5d9e5d)'
  if (t === 'down') return 'var(--red, #b9524c)'
  return 'var(--muted)'
}
</script>

<template>
  <section class="episodes-section">
    <div class="ep-header">
      <div class="ep-title-row">
        <span class="section-header">episodes</span>
        <span v-if="stats" class="ep-count">{{ stats.total }}</span>
      </div>
      <div v-if="stats" class="ep-stats">
        <div class="ep-stat-item">
          <span class="ep-stat-label">best</span>
          <span class="ep-stat-val ep-stat-best">{{ fmt(stats.best) }}</span>
        </div>
        <div class="ep-stat-item">
          <span class="ep-stat-label">avg</span>
          <span class="ep-stat-val">{{ fmt(stats.meanScore) }}</span>
        </div>
        <div class="ep-stat-item">
          <span class="ep-stat-label">last 5</span>
          <span class="ep-stat-val" :style="{ color: trendColor(stats.trend) }">
            {{ fmt(stats.last5Avg) }} {{ trendIcon(stats.trend) }}
          </span>
        </div>
        <div class="ep-stat-item">
          <span class="ep-stat-label">avg len</span>
          <span class="ep-stat-val ep-stat-muted">{{ fmt(stats.meanLength) }}</span>
        </div>
      </div>
    </div>

    <!-- Checkpoint info bar -->
    <div v-if="bestCheckpointEp && bestCkptLabel" class="ep-ckpt-bar">
      <span class="ep-ckpt-icon">★</span>
      <span class="ep-ckpt-text">
        best ckpt: ep {{ bestCheckpointEp.episode }} &middot; score {{ fmt(bestCheckpointEp.score) }}
      </span>
      <span v-if="hasCheckpoint" class="ep-ckpt-badge">saved</span>
    </div>

    <div class="episodes-scroll">
      <div v-if="recentEpisodes.length === 0" class="ep-empty">
        No episodes yet
      </div>
      <div v-else class="ep-list">
        <div class="ep-row ep-row-head">
          <span class="ep-cell ep-num">#</span>
          <span class="ep-cell ep-step">step</span>
          <span class="ep-cell ep-ret">return</span>
          <span class="ep-cell ep-score-hd">score</span>
        </div>
        <div
          v-for="(e, idx) in recentEpisodes"
          :key="e.episode as number"
          class="ep-row"
          :class="[rowClass(e as Record<string, unknown>), { 'ep-newest': idx === 0 }]"
        >
          <span class="ep-cell ep-num">{{ e.episode ?? '' }}</span>
          <span class="ep-cell ep-step">{{ e.step != null ? fmt(e.step) : '' }}</span>
          <span class="ep-cell ep-ret">{{ fmt(e.return) }}</span>
          <span class="ep-cell ep-score" :style="{ color: scoreColor((e.score as number) ?? 0) }">
            {{ fmt(e.score) }}
          </span>
        </div>
      </div>
    </div>
  </section>
</template>
