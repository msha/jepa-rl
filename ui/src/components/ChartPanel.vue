<script setup lang="ts">
import { ref, onMounted, toRef } from 'vue'
import { useChart } from '../composables/useChart'

const props = defineProps<{
  label: string
  points: [number, number][]
  color: string
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)
const { draw } = useChart(canvasRef, toRef(props, 'points'), toRef(props, 'color'))

onMounted(() => {
  draw()
  // Redraw on resize
  const observer = new ResizeObserver(() => draw())
  if (canvasRef.value) observer.observe(canvasRef.value)
})
</script>

<template>
  <div class="chart-panel">
    <div class="section-label">{{ label }}</div>
    <canvas ref="canvasRef"></canvas>
  </div>
</template>
