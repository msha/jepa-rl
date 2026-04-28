<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ episodes: Record<string, unknown>[] }>()

const recentEpisodes = computed(() => {
  return (props.episodes || []).slice(-20).reverse()
})

function fmt(v: unknown): string {
  if (v == null || (typeof v === 'number' && Number.isNaN(v))) return '—'
  if (typeof v === 'number') {
    if (Math.abs(v) >= 1000) return v.toFixed(0)
    if (Math.abs(v) >= 10) return v.toFixed(2)
    return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
  }
  return String(v)
}
</script>

<template>
  <section class="episodes-section">
    <div class="section-header" title="Recent training episodes with their scores">episodes</div>
    <div class="episodes-scroll">
      <table>
        <thead>
          <tr><th>ep</th><th>step</th><th>return</th><th>score</th></tr>
        </thead>
        <tbody>
          <tr v-for="e in recentEpisodes" :key="(e.episode as number)">
            <td>{{ e.episode ?? '' }}</td>
            <td>{{ e.step ?? '' }}</td>
            <td>{{ fmt(e.return) }}</td>
            <td>{{ fmt(e.score) }}</td>
          </tr>
        </tbody>
      </table>
    </div>
  </section>
</template>
