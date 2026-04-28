import re

with open("ui/src/assets/components.css", "r") as f:
    content = f.read()

train_controls_idx = content.find("/* Train controls */")
if train_controls_idx != -1:
    content = content[:train_controls_idx]

new_styles = """
/* --- New Train Controls UI --- */
.tc-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 12px;
  background: var(--bg);
}

.tc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.tc-card.status-card {
  border-color: var(--accent);
  background: rgba(196,145,82,0.03);
}

.tc-card-header {
  display: flex;
  flex-direction: column;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 10px;
  margin-bottom: 4px;
}

.tc-card-header h3 {
  margin: 0;
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent);
}

.tc-help-text {
  margin: 0;
  font-size: 11px;
  color: var(--muted);
  line-height: 1.4;
}

.tc-grid-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
}

.tc-field-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.tc-field-group label {
  font-size: 10px;
  font-weight: 600;
  color: var(--text);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.tc-field-desc {
  font-size: 9px;
  color: var(--muted);
  line-height: 1.2;
}

.tc-field-group input, .tc-field-group select {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--text);
  padding: 6px 8px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 12px;
  width: 100%;
  box-sizing: border-box;
  transition: all 0.2s;
}

.tc-field-group input:focus, .tc-field-group select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(196,145,82,0.3);
}

.tc-card-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-top: 1px solid var(--border);
  padding-top: 12px;
  margin-top: 4px;
}

.tc-checkbox-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text);
  cursor: pointer;
}

.tc-checkbox-label input {
  accent-color: var(--accent);
  cursor: pointer;
  width: 14px;
  height: 14px;
}

.tc-buttons {
  display: flex;
  gap: 8px;
}

.btn-primary {
  background: rgba(196,145,82,0.15);
  border: 1px solid var(--accent);
  color: var(--accent);
  border-radius: 4px;
  padding: 6px 16px;
  font-family: 'Outfit', sans-serif;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary:hover {
  background: rgba(196,145,82,0.25);
  box-shadow: 0 0 8px rgba(196,145,82,0.4);
}

.btn-secondary {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 4px;
  padding: 6px 12px;
  font-family: 'Outfit', sans-serif;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-secondary:hover {
  border-color: var(--text);
}

.btn-danger {
  background: rgba(185,82,76,0.1);
  border: 1px solid var(--red);
  color: var(--red);
  border-radius: 4px;
  padding: 6px 16px;
  font-family: 'Outfit', sans-serif;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-danger:hover {
  background: rgba(185,82,76,0.2);
}

.tc-status-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.tc-status-text {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text);
}

.tc-elapsed-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
}

.tc-elapsed { color: var(--text); }
.tc-eta { color: var(--accent); }

.tc-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.tc-dot.idle { background: var(--muted); }
.tc-dot.running { background: #5d9e5d; animation: pulse 1.5s infinite; }
.tc-dot.evaluating { background: var(--accent); animation: pulse 1.5s infinite; }
.tc-dot.stopped { background: var(--muted); }
.tc-dot.error { background: var(--red); }

.tc-status-detail {
  color: var(--muted);
  font-family: 'IBM Plex Mono', monospace;
}

.tc-progress-track {
  height: 4px;
  background: var(--bg);
  border-radius: 2px;
  overflow: hidden;
  margin-top: 4px;
}
.tc-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #5d9e5d, var(--accent));
  border-radius: 2px;
  transition: width 0.3s ease;
}

.tc-error {
  color: var(--red);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  padding: 6px;
  background: rgba(185,82,76,0.1);
  border-left: 2px solid var(--red);
  border-radius: 2px;
}

.tc-status-highlight {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: var(--accent);
  padding: 6px;
  background: rgba(196,145,82,0.1);
  border-left: 2px solid var(--accent);
  border-radius: 2px;
}

.tc-tools-grid {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.tc-tool-item {
  padding-bottom: 12px;
  border-bottom: 1px dashed var(--border);
}
.tc-tool-item:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.tc-badge-ok { color: #5d9e5d; font-family: 'IBM Plex Mono', monospace; }
.tc-badge-err { color: var(--red); font-family: 'IBM Plex Mono', monospace; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
"""

with open("ui/src/assets/components.css", "w") as f:
    f.write(content + new_styles)
