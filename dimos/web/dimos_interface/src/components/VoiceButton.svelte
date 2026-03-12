<!--
 Copyright 2025 Dimensional Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { theme } from '../stores/theme';
  import { connectTextStream } from '../stores/stream';

  const dispatch = createEventDispatcher();

  const getServerUrl = () => {
    const hostname = window.location.hostname;
    return `http://${hostname}:5555`;
  };

  let isRecording = false;
  let mediaRecorder: MediaRecorder | null = null;
  let chunks: Blob[] = [];
  let isProcessing = false;

  async function toggleRecording() {
    if (isRecording && mediaRecorder) {
      mediaRecorder.stop();
      isRecording = false;
    } else {
      try {
        if (!mediaRecorder) {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream);

          mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

          mediaRecorder.onstop = async () => {
            isProcessing = true;
            const blob = new Blob(chunks, { type: 'audio/webm' });
            chunks = [];

            const formData = new FormData();
            formData.append('file', blob, 'recording.webm');

            try {
              const res = await fetch(`${getServerUrl()}/upload_audio`, {
                method: 'POST',
                body: formData
              });

              const json = await res.json();

              if (json.success) {
                connectTextStream('agent_responses');
                dispatch('voiceCommand', { success: true });
              } else {
                dispatch('voiceCommand', {
                  success: false,
                  error: json.message
                });
              }
            } catch (err) {
              dispatch('voiceCommand', {
                success: false,
                error: err instanceof Error ? err.message : 'Upload failed'
              });
            } finally {
              isProcessing = false;
            }
          };
        }

        mediaRecorder.start();
        isRecording = true;
      } catch (err) {
        dispatch('voiceCommand', {
          success: false,
          error: 'Microphone access denied'
        });
      }
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if ((event.ctrlKey || event.metaKey) && event.key === 'm') {
      event.preventDefault();
      toggleRecording();
    }
  }
</script>

<svelte:window on:keydown={handleKeyPress} />

<div class="voice-container">
  <!-- Outer ring -->
  <div class="voice-ring" class:active={isRecording}></div>
  <!-- Middle ring -->
  <div class="voice-ring-inner" class:active={isRecording}></div>

  <button
    class="voice-btn"
    class:recording={isRecording}
    class:processing={isProcessing}
    on:click={toggleRecording}
    disabled={isProcessing}
    title={isRecording ? 'Stop recording (Ctrl+M)' : 'Start voice command (Ctrl+M)'}
  >
    {#if isProcessing}
      <svg class="icon spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"/>
      </svg>
    {:else if isRecording}
      <svg class="icon" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="6" width="12" height="12" rx="2"/>
      </svg>
    {:else}
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="9" y="2" width="6" height="12" rx="3"/>
        <path d="M5 10a7 7 0 0 0 14 0" stroke-linecap="round"/>
        <line x1="12" y1="17" x2="12" y2="22" stroke-linecap="round"/>
        <line x1="8" y1="22" x2="16" y2="22" stroke-linecap="round"/>
      </svg>
    {/if}
  </button>

  <span class="voice-label" class:recording={isRecording}>
    {isProcessing ? 'PROCESSING' : isRecording ? 'RECORDING' : 'VOICE'}
  </span>
</div>

<style>
  .voice-container {
    position: fixed;
    bottom: 28px;
    right: 28px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    z-index: 1000;
  }

  /* Outer ring */
  .voice-ring {
    position: absolute;
    width: 80px;
    height: 80px;
    border-radius: 50%;
    border: 1px solid rgba(0, 240, 255, 0.15);
    top: 50%;
    left: 50%;
    transform: translate(-50%, calc(-50% - 10px));
    transition: all 0.4s ease;
    pointer-events: none;
  }

  .voice-ring.active {
    width: 100px;
    height: 100px;
    border-color: var(--hud-red);
    animation: ring-pulse 1.5s infinite;
  }

  /* Inner ring */
  .voice-ring-inner {
    position: absolute;
    width: 70px;
    height: 70px;
    border-radius: 50%;
    border: 1px dashed rgba(0, 240, 255, 0.1);
    top: 50%;
    left: 50%;
    transform: translate(-50%, calc(-50% - 10px));
    transition: all 0.4s ease;
    pointer-events: none;
    animation: slow-spin 20s linear infinite;
  }

  .voice-ring-inner.active {
    border-color: rgba(255, 34, 68, 0.4);
    animation: slow-spin 4s linear infinite;
  }

  .voice-btn {
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: rgba(0, 240, 255, 0.06);
    border: 2px solid rgba(0, 240, 255, 0.4);
    color: var(--hud-cyan);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    box-shadow:
      0 0 15px rgba(0, 240, 255, 0.1),
      inset 0 0 15px rgba(0, 240, 255, 0.05);
    backdrop-filter: blur(8px);
  }

  .voice-btn:hover:not(:disabled) {
    background: rgba(0, 240, 255, 0.15);
    border-color: var(--hud-cyan);
    box-shadow:
      0 0 25px rgba(0, 240, 255, 0.3),
      inset 0 0 20px rgba(0, 240, 255, 0.1);
    transform: scale(1.08);
  }

  .voice-btn:active:not(:disabled) {
    transform: scale(0.95);
  }

  .voice-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .voice-btn.recording {
    border-color: var(--hud-red);
    color: var(--hud-red);
    background: rgba(255, 34, 68, 0.1);
    box-shadow:
      0 0 20px rgba(255, 34, 68, 0.3),
      inset 0 0 15px rgba(255, 34, 68, 0.1);
    animation: btn-pulse 1.5s infinite;
  }

  .voice-btn.recording:hover {
    background: rgba(255, 34, 68, 0.25);
    box-shadow: 0 0 30px rgba(255, 34, 68, 0.5);
  }

  .voice-btn.processing {
    border-style: dashed;
    border-color: var(--hud-amber);
    color: var(--hud-amber);
  }

  .icon {
    width: 22px;
    height: 22px;
  }

  .spin {
    animation: spinner 0.8s linear infinite;
  }

  .voice-label {
    font-family: 'Orbitron', monospace;
    font-size: 8px;
    letter-spacing: 3px;
    color: rgba(0, 240, 255, 0.4);
    text-transform: uppercase;
  }

  .voice-label.recording {
    color: var(--hud-red);
    animation: blink 1s infinite;
  }

  @keyframes ring-pulse {
    0%, 100% {
      transform: translate(-50%, calc(-50% - 10px)) scale(1);
      opacity: 0.6;
    }
    50% {
      transform: translate(-50%, calc(-50% - 10px)) scale(1.1);
      opacity: 1;
    }
  }

  @keyframes slow-spin {
    from { transform: translate(-50%, calc(-50% - 10px)) rotate(0deg); }
    to { transform: translate(-50%, calc(-50% - 10px)) rotate(360deg); }
  }

  @keyframes btn-pulse {
    0%, 100% { box-shadow: 0 0 15px rgba(255, 34, 68, 0.2); }
    50% { box-shadow: 0 0 30px rgba(255, 34, 68, 0.5); }
  }

  @keyframes spinner {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  @media (max-width: 640px) {
    .voice-btn {
      width: 48px;
      height: 48px;
    }
    .voice-ring { width: 68px; height: 68px; }
    .voice-ring-inner { width: 60px; height: 60px; }
    .voice-container { bottom: 16px; right: 16px; }
    .icon { width: 18px; height: 18px; }
  }
</style>
