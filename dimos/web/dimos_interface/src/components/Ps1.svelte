<script>
  import { theme } from '../stores/theme';
  let hostname = window.location.hostname;
</script>

<div class="ps1">
  <span class="ps1-bracket">[</span>
  <span class="ps1-user" style={`color: ${$theme.yellow};`}>nightwatch</span>
  <span class="ps1-at" style={`color: ${$theme.white}`}>@</span>
  <span class="ps1-host" style={`color: ${$theme.green}`}>{hostname}</span>
  <span class="ps1-bracket">]</span>
  <span class="ps1-arrow">&gt;</span>
</div>

<style>
  .ps1 {
    display: flex;
    align-items: center;
    font-weight: bold;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .ps1-bracket {
    color: var(--hud-cyan);
    opacity: 0.4;
  }

  .ps1-user {
    text-shadow: 0 0 8px currentColor;
  }

  .ps1-at {
    opacity: 0.5;
    margin: 0 1px;
  }

  .ps1-host {
    text-shadow: 0 0 8px currentColor;
  }

  .ps1-arrow {
    color: var(--hud-cyan);
    margin-left: 4px;
    animation: blink-cursor 1s step-end infinite;
    text-shadow: 0 0 6px var(--hud-cyan);
  }

  @keyframes blink-cursor {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
  }
</style>
