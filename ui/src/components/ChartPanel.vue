<script setup lang="ts">
import { ref, computed, onMounted, toRef } from 'vue'
import { useChart } from '../composables/useChart'

const props = defineProps<{
  label: string
  points: [number, number][]
  color: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const { draw } = useChart(canvasRef, toRef(props, 'points'), toRef(props, 'color'))

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
    <canvas ref="canvasRef"></canvas>
  </div>
</template>
