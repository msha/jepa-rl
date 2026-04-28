import { onMounted, onUnmounted, ref } from 'vue'

export function usePolling(fn: () => Promise<void>, intervalMs: number) {
  const paused = ref(false)
  let timer: ReturnType<typeof setTimeout> | null = null

  function loop() {
    if (timer !== null) clearTimeout(timer)
    timer = setTimeout(async () => {
      if (!paused.value) {
        try { await fn() } catch { /* swallow poll errors */ }
      }
      loop()
    }, intervalMs)
  }

  function stop() {
    if (timer !== null) { clearTimeout(timer); timer = null }
  }

  onMounted(loop)
  onUnmounted(stop)

  return { paused, stop }
}
