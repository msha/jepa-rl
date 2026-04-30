<script setup lang="ts">
import AppHeader from './components/AppHeader.vue'
import MetricsGrid from './components/MetricsGrid.vue'
import ChartPanel from './components/ChartPanel.vue'
import GameSection from './components/GameSection.vue'
import GameSettings from './components/GameSettings.vue'
import HighScoreBoard from './components/HighScoreBoard.vue'
import TrainControls from './components/TrainControls.vue'
import ConfigPanel from './components/ConfigPanel.vue'
import EpisodesTable from './components/EpisodesTable.vue'

import { useTrainingStore } from './stores/training'
import { useConfigStore } from './stores/config'
import { useRunsStore } from './stores/runs'
import { usePolling } from './composables/usePolling'
import { computed, onMounted, ref } from 'vue'

const training = useTrainingStore()
const config = useConfigStore()
const runs = useRunsStore()
const highscores = ref<{ score: number; player: string }[]>([])
const playerName = ref('')

const isWorldTraining = computed(() =>
  training.isWorldTraining ||
  (training.worldJob?.status === 'completed') ||
  (training.worldJob?.status === 'error')
)

usePolling(() => training.refresh(), 300)
usePolling(() => runs.loadRuns(), 5000)

onMounted(async () => {
  await config.loadConfigs()
  await training.refresh()
  await runs.loadRuns()
  const activeRun = training.runDir?.name
  if (!runs.selectedRun && activeRun && runs.runs.some(r => r.name === activeRun)) {
    await runs.loadRunDetail(activeRun)
    await training.refresh()
  }
})
</script>

<template>
  <AppHeader :run-name="training.runDir?.name ?? ''" :status-text="training.headerStatus" />
  <main>
    <div class="col col-left">
      <MetricsGrid :summary="training.summary" :latest-step="training.latestStep" :eval-result="training.evalResult" :world-job="training.worldJob" :collect-job="training.collectJob" />
      <div class="charts">
        <ChartPanel v-if="isWorldTraining" label="loss" :points="training.chartPoints('loss')" color="#4e89ba" />
        <ChartPanel v-if="isWorldTraining" label="pred loss" :points="training.chartPoints('prediction_loss')" color="#b9524c" />
        <ChartPanel v-if="!isWorldTraining" label="score" :points="training.chartPoints('score')" color="#5d9e5d" />
        <ChartPanel v-if="!isWorldTraining" label="loss" :points="training.chartPoints('loss')" color="#4e89ba" />
        <ChartPanel v-if="!isWorldTraining" label="epsilon" :points="training.chartPoints('epsilon')" color="#bfa03e" />
        <ChartPanel v-if="!isWorldTraining" label="td error" :points="training.chartPoints('td')" color="#b9524c" />
      </div>
      <EpisodesTable
        :episodes="training.episodes"
        :summary="training.summary"
        :checkpoints="training.runDir?.checkpoints ?? []"
        :has-checkpoint="training.runDir?.has_checkpoint ?? false"
      />
    </div>
    <div class="col col-center">
      <GameSection
        :job="training.job"
        :eval-job="training.evalJob"
        :collect-job="training.collectJob"
        :reset-key="training.resetKey"
        :action-keys="training.actionKeys"
        :steps="training.steps"
        :player-name="playerName"
        :model-info="training.modelInfo"
        @highscores="highscores = $event"
      />
      <GameSettings :settings="training.gameSettings" />
      <HighScoreBoard :highscores="highscores" @update:player-name="playerName = $event" />
    </div>
    <div class="col col-right">
      <TrainControls :job="training.job" :eval-job="training.evalJob" :summary="training.summary" :latest-step="training.latestStep" />
      <ConfigPanel />
    </div>
  </main>
</template>
