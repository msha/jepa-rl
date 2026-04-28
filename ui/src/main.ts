import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'

import './assets/variables.css'
import './assets/base.css'
import './assets/layout.css'
import './assets/components.css'

createApp(App).use(createPinia()).mount('#app')
