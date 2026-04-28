<script setup lang="ts">
import AppHeader from './components/AppHeader.vue'
import ControlsBar from './components/ControlsBar.vue'
import MetricsGrid from './components/MetricsGrid.vue'
import ChartPanel from './components/ChartPanel.vue'
import GameSection from './components/GameSection.vue'
import GameSettings from './components/GameSettings.vue'
import RunSelector from './components/RunSelector.vue'
import RunInfoBar from './components/RunInfoBar.vue'
import TrainControls from './components/TrainControls.vue'
import ConfigPanel from './components/ConfigPanel.vue'
import EpisodesTable from './components/EpisodesTable.vue'

import { useTrainingStore } from './stores/training'
import { useConfigStore } from './stores/config'
import { useRunsStore } from './stores/runs'
import { usePolling } from './composables/usePolling'
import { onMounted } from 'vue'

const training = useTrainingStore()
const config = useConfigStore()
const runs = useRunsStore()

usePolling(() => training.refresh(), 300)
usePolling(() => runs.loadRuns(), 5000)

onMounted(() => {
  config.loadConfigs()
  runs.loadRuns()
  training.refresh()
})
</script>

<template>
  <AppHeader :run-name="training.runDir?.name ?? ''" :status-text="training.headerStatus" />
  <ControlsBar />
  <main>
    <div class="col col-left">
      <MetricsGrid :summary="training.summary" :latest-step="training.latestStep" :eval-result="training.evalResult" />
      <div class="charts">
        <ChartPanel label="score" :points="training.chartPoints('score')" color="#5d9e5d" />
        <ChartPanel label="loss" :points="training.chartPoints('loss')" color="#4e89ba" />
        <ChartPanel label="epsilon" :points="training.chartPoints('epsilon')" color="#bfa03e" />
        <ChartPanel label="td error" :points="training.chartPoints('td')" color="#b9524c" />
      </div>
      <EpisodesTable :episodes="training.episodes" />
    </div>
    <div class="col col-center">
      <GameSection
        :job="training.job"
        :eval-job="training.evalJob"
        :reset-key="training.resetKey"
        :action-keys="training.actionKeys"
        :steps="training.steps"
      />
      <GameSettings :settings="training.gameSettings" />
    </div>
    <div class="col col-right">
      <RunSelector />
      <RunInfoBar :job="training.job" :eval-job="training.evalJob" :summary="training.summary" :latest-step="training.latestStep" />
      <TrainControls />
      <ConfigPanel />
    </div>
  </main>
</template>
