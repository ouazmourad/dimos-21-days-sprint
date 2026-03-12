<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { streamStore } from '../stores/stream';

  let errorMessages: Record<string, string | null> = {};
  let retryCount: Record<string, number> = {};
  let retryTimers: Record<string, number | null> = {};
  let timestamps: Record<string, number> = {};
  const TOTAL_TIMEOUT = 120000;
  const RETRY_INTERVAL = 2000;
  const MAX_RETRIES = Math.floor(TOTAL_TIMEOUT / RETRY_INTERVAL);

  const initialState = {
    isVisible: false,
    url: null,
    streamKeys: [],
    availableStreams: []
  };

  function clearRetryTimer(streamKey: string) {
    if (retryTimers[streamKey] !== null) {
      clearTimeout(retryTimers[streamKey]);
      retryTimers[streamKey] = null;
    }
  }

  function retryConnection(streamKey: string) {
    if (!retryCount[streamKey]) retryCount[streamKey] = 0;

    if (retryCount[streamKey] < MAX_RETRIES) {
      retryCount[streamKey]++;
      const timeLeft = TOTAL_TIMEOUT - (retryCount[streamKey] * RETRY_INTERVAL);
      errorMessages[streamKey] = `RECONNECTING ${retryCount[streamKey]}/${MAX_RETRIES} [${Math.ceil(timeLeft / 1000)}s]`;

      timestamps[streamKey] = Date.now();

      clearRetryTimer(streamKey);
      retryTimers[streamKey] = setTimeout(() => retryConnection(streamKey), RETRY_INTERVAL);
    } else {
      errorMessages[streamKey] = 'FEED OFFLINE — Check Robot() connection';
    }
  }

  function handleError(streamKey: string) {
    if (!retryCount[streamKey] || retryCount[streamKey] === 0) {
      retryConnection(streamKey);
    }
  }

  function handleLoad(streamKey: string) {
    errorMessages[streamKey] = null;
    retryCount[streamKey] = 0;
    clearRetryTimer(streamKey);
  }

  function stopStream() {
    Object.keys(retryTimers).forEach(key => clearRetryTimer(key));
    streamStore.set(initialState);
  }

  $: if ($streamStore.url && $streamStore.streamKeys) {
    $streamStore.streamKeys.forEach(key => {
      errorMessages[key] = null;
      retryCount[key] = 0;
      clearRetryTimer(key);
      timestamps[key] = Date.now();
    });
  }

  onDestroy(() => {
    Object.keys(retryTimers).forEach(key => clearRetryTimer(key));
  });

  $: streamUrls = $streamStore.streamKeys.map(key => ({
    key,
    url: $streamStore.url ? `${$streamStore.url}/video_feed/${key}?t=${timestamps[key] || Date.now()}` : null
  }));

  $: gridCols = Math.ceil(Math.sqrt($streamStore.streamKeys.length));
  $: gridRows = Math.ceil($streamStore.streamKeys.length / gridCols);
</script>

<div class="stream-viewer" class:visible={$streamStore.isVisible}>
  <div class="stream-panel" style="--grid-cols: {gridCols}; --grid-rows: {gridRows};">
    <!-- HUD header -->
    <div class="stream-header">
      <span class="stream-header-dot"></span>
      <span class="stream-header-title">LIVE FEED</span>
      <button class="close-btn" on:click={stopStream}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>

    {#if $streamStore.isVisible}
      <div class="stream-grid">
        {#each streamUrls as {key, url}}
          <div class="stream-cell">
            <!-- Corner accents -->
            <div class="cell-corner tl"></div>
            <div class="cell-corner tr"></div>
            <div class="cell-corner bl"></div>
            <div class="cell-corner br"></div>

            {#if url}
              <img
                src={url}
                alt={`Feed: ${key}`}
                on:error={() => handleError(key)}
                on:load={() => handleLoad(key)}
              />
            {/if}
            <div class="cell-label">{key.toUpperCase()}</div>
            {#if errorMessages[key]}
              <div class="error-overlay">
                <span class="error-text">{errorMessages[key]}</span>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .stream-viewer {
    position: fixed;
    top: 12px;
    right: 12px;
    z-index: 1000;
    display: none;
  }

  .visible {
    display: block;
  }

  .stream-panel {
    background: rgba(10, 10, 15, 0.95);
    border: 1px solid rgba(0, 240, 255, 0.3);
    box-shadow:
      0 0 20px rgba(0, 240, 255, 0.1),
      inset 0 0 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(12px);
    overflow: hidden;
  }

  .stream-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-bottom: 1px solid rgba(0, 240, 255, 0.15);
    background: rgba(0, 240, 255, 0.04);
  }

  .stream-header-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ff2244;
    box-shadow: 0 0 6px #ff2244;
    animation: rec-blink 1s infinite;
  }

  @keyframes rec-blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .stream-header-title {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 3px;
    color: rgba(0, 240, 255, 0.6);
    flex: 1;
  }

  .close-btn {
    background: none;
    border: 1px solid rgba(0, 240, 255, 0.2);
    color: rgba(0, 240, 255, 0.5);
    cursor: pointer;
    padding: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
  }

  .close-btn:hover {
    border-color: var(--hud-red);
    color: var(--hud-red);
    background: rgba(255, 34, 68, 0.1);
  }

  .stream-grid {
    display: grid;
    grid-template-columns: repeat(var(--grid-cols), 1fr);
    gap: 4px;
    padding: 8px;
    max-width: 85vw;
  }

  .stream-cell {
    position: relative;
    aspect-ratio: 4/3;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.6);
    border: 1px solid rgba(0, 240, 255, 0.08);
    min-width: 280px;
    overflow: hidden;
  }

  .stream-cell img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }

  /* Corner accents on each cell */
  .cell-corner {
    position: absolute;
    width: 10px;
    height: 10px;
    z-index: 2;
    pointer-events: none;
  }
  .cell-corner.tl { top: 0; left: 0; border-top: 1px solid var(--hud-cyan); border-left: 1px solid var(--hud-cyan); }
  .cell-corner.tr { top: 0; right: 0; border-top: 1px solid var(--hud-cyan); border-right: 1px solid var(--hud-cyan); }
  .cell-corner.bl { bottom: 0; left: 0; border-bottom: 1px solid var(--hud-cyan); border-left: 1px solid var(--hud-cyan); }
  .cell-corner.br { bottom: 0; right: 0; border-bottom: 1px solid var(--hud-cyan); border-right: 1px solid var(--hud-cyan); }

  .cell-label {
    position: absolute;
    bottom: 4px;
    left: 6px;
    font-family: 'Orbitron', monospace;
    font-size: 8px;
    letter-spacing: 2px;
    color: rgba(0, 240, 255, 0.4);
    z-index: 3;
  }

  .error-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.8);
  }

  .error-text {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--hud-red);
    text-align: center;
    padding: 12px;
  }
</style>
