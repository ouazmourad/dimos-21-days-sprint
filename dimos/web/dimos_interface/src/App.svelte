<script lang="ts">
  import Ps1 from './components/Ps1.svelte';
  import Input from './components/Input.svelte';
  import History from './components/History.svelte';
  import StreamViewer from './components/StreamViewer.svelte';
  import VoiceButton from './components/VoiceButton.svelte';
  import { theme } from './stores/theme';
  import { history } from './stores/history';
  import { onMount } from 'svelte';

  let currentTime = '';
  let uptimeSeconds = 0;

  const handleVoiceCommand = async (event: CustomEvent) => {
    if (event.detail.success) {
      history.update(h => [...h, {
        command: '[voice command]',
        outputs: ['Processing voice command...']
      }]);
    } else {
      history.update(h => [...h, {
        command: '[voice command]',
        outputs: [`Error: ${event.detail.error}`]
      }]);
    }
  };

  onMount(() => {
    const timer = setInterval(() => {
      const now = new Date();
      currentTime = now.toLocaleTimeString('en-US', { hour12: false });
      uptimeSeconds++;
    }, 1000);
    return () => clearInterval(timer);
  });

  function formatUptime(s: number): string {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
  }
</script>

<svelte:head>
  {#if import.meta.env.VITE_TRACKING_ENABLED === 'true'}
    <script
      async
      defer
      data-website-id={import.meta.env.VITE_TRACKING_SITE_ID}
      src={import.meta.env.VITE_TRACKING_URL}
    ></script>
  {/if}
</svelte:head>

<div class="hud-container">
  <!-- Top HUD bar -->
  <header class="hud-header">
    <div class="hud-header-left">
      <span class="hud-logo">NIGHTWATCH</span>
      <span class="hud-divider">|</span>
      <span class="hud-label">DimOS v2.0</span>
    </div>
    <div class="hud-header-center">
      <span class="hud-status-dot"></span>
      <span class="hud-label">SYSTEM ONLINE</span>
    </div>
    <div class="hud-header-right">
      <span class="hud-label">UPTIME {formatUptime(uptimeSeconds)}</span>
      <span class="hud-divider">|</span>
      <span class="hud-time">{currentTime}</span>
    </div>
  </header>

  <!-- Main terminal area -->
  <main
    class="hud-terminal"
    style={`background-color: ${$theme.background}ee; color: ${$theme.foreground};`}
  >
    <!-- Corner decorations -->
    <div class="corner corner-tl"></div>
    <div class="corner corner-tr"></div>
    <div class="corner corner-bl"></div>
    <div class="corner corner-br"></div>

    <div class="terminal-content">
      <StreamViewer />
      <History />

      <div class="input-row">
        <Ps1 />
        <Input />
      </div>
    </div>
  </main>

  <!-- Bottom HUD bar -->
  <footer class="hud-footer">
    <span class="hud-label">CTRL+M VOICE</span>
    <span class="hud-divider">|</span>
    <span class="hud-label">TAB AUTOCOMPLETE</span>
    <span class="hud-divider">|</span>
    <span class="hud-label">CTRL+L CLEAR</span>
    <span class="hud-footer-right">
      <span class="hud-label">UNITREE GO2</span>
      <span class="hud-status-dot small"></span>
    </span>
  </footer>
</div>

<VoiceButton on:voiceCommand={handleVoiceCommand} />

<style>
  .hud-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 8px;
    gap: 4px;
    position: relative;
    z-index: 1;
  }

  /* ── Header ── */
  .hud-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 16px;
    border: 1px solid var(--hud-border);
    border-bottom: 1px solid var(--hud-cyan);
    background: linear-gradient(180deg, rgba(0, 240, 255, 0.06) 0%, transparent 100%);
    flex-shrink: 0;
  }

  .hud-header-left,
  .hud-header-center,
  .hud-header-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .hud-logo {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 14px;
    letter-spacing: 4px;
    color: var(--hud-cyan);
    text-shadow: 0 0 10px var(--hud-cyan), 0 0 30px rgba(0, 240, 255, 0.3);
  }

  .hud-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(0, 240, 255, 0.5);
  }

  .hud-time {
    font-family: 'Orbitron', monospace;
    font-size: 12px;
    color: var(--hud-cyan);
    letter-spacing: 2px;
  }

  .hud-divider {
    color: rgba(0, 240, 255, 0.2);
    font-size: 12px;
  }

  .hud-status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00ff88;
    box-shadow: 0 0 6px #00ff88, 0 0 12px rgba(0, 255, 136, 0.4);
    animation: pulse-dot 2s infinite;
  }

  .hud-status-dot.small {
    width: 4px;
    height: 4px;
  }

  @keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* ── Terminal ── */
  .hud-terminal {
    flex: 1;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--hud-border);
    box-shadow:
      inset 0 0 30px rgba(0, 240, 255, 0.03),
      0 0 1px var(--hud-cyan);
  }

  .terminal-content {
    height: 100%;
    overflow-y: auto;
    padding: 16px 20px;
    padding-bottom: 8px;
  }

  .input-row {
    display: flex;
    flex-direction: row;
    align-items: center;
  }

  @media (max-width: 768px) {
    .input-row {
      flex-direction: column;
      align-items: flex-start;
    }
  }

  /* Corner decorations */
  .corner {
    position: absolute;
    width: 16px;
    height: 16px;
    z-index: 2;
  }

  .corner-tl {
    top: 0; left: 0;
    border-top: 2px solid var(--hud-cyan);
    border-left: 2px solid var(--hud-cyan);
  }
  .corner-tr {
    top: 0; right: 0;
    border-top: 2px solid var(--hud-cyan);
    border-right: 2px solid var(--hud-cyan);
  }
  .corner-bl {
    bottom: 0; left: 0;
    border-bottom: 2px solid var(--hud-cyan);
    border-left: 2px solid var(--hud-cyan);
  }
  .corner-br {
    bottom: 0; right: 0;
    border-bottom: 2px solid var(--hud-cyan);
    border-right: 2px solid var(--hud-cyan);
  }

  /* ── Footer ── */
  .hud-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 16px;
    border: 1px solid var(--hud-border);
    border-top: 1px solid rgba(0, 240, 255, 0.1);
    background: linear-gradient(0deg, rgba(0, 240, 255, 0.04) 0%, transparent 100%);
    flex-shrink: 0;
  }

  .hud-footer-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
  }
</style>
