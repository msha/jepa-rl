<script setup lang="ts">
import { ref, computed, onMounted, toRef } from 'vue'
import { useChart } from '../composables/useChart'

const props = defineProps<{
  label: string
  points: [number, number][]
  color: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const { draw, minYPx, maxYPx, minVal, maxVal } = useChart(canvasRef, toRef(props, 'points'), toRef(props, 'color'))

const currentVal = computed(() => {
  const pts = props.points
  if (!pts.length) return null
  return pts[pts.length - 1][1]
})

function fmt(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return ''
  if (Math.abs(v) >= 1000) return v.toFixed(0)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')
}

const labelMaxTop = computed(() => Math.max(6, Math.min(74, maxYPx.value)))
const labelMinTop = computed(() => Math.max(6, Math.min(74, minYPx.value)))

onMounted(() => {
  draw()
  const observer = new ResizeObserver(() => draw())
  if (canvasRef.value) observer.observe(canvasRef.value)
})
</script>

<template>
  <div class="chart-panel">
    <div class="section-label">
      <span>{{ label }}</span>
      <span v-if="currentVal != null" class="chart-value" :style="{ color }">{{ fmt(currentVal) }}</span>
    </div>
    <div class="chart-row">
      <canvas ref="canvasRef"></canvas>
      <div class="chart-labels">
        <span v-if="maxVal != null" class="chart-ext" :style="{ top: labelMaxTop + 'px', color }">▲ {{ fmt(maxVal) }}</span>
        <span v-if="minVal != null && minVal !== maxVal" class="chart-ext" :style="{ top: labelMinTop + 'px', color }">▼ {{ fmt(minVal) }}</span>
      </div>
    </div>
  </div>
</template>
