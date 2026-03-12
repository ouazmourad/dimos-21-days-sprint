<script lang="ts">
  import { afterUpdate, onMount } from 'svelte';
  import { history } from '../stores/history';
  import { theme } from '../stores/theme';
  import { commands } from '../utils/commands';
  import { track } from '../utils/tracking';
  import { connectTextStream } from '../stores/stream';

  let command = '';
  let historyIndex = -1;
  let input: HTMLInputElement;

  onMount(() => {
    input.focus();

    if ($history.length === 0) {
      const command = commands['banner'] as () => string;

      if (command) {
        const output = command();
        $history = [...$history, { command: 'banner', outputs: [output] }];
      }
    }
  });

  afterUpdate(() => {
    input.scrollIntoView({ behavior: 'smooth', block: 'end' });
  });

  const handleKeyDown = async (event: KeyboardEvent) => {
    if (event.key === 'Enter') {
      await executeCommand();
    } else if (event.key === 'ArrowUp') {
      if (historyIndex < $history.length - 1) {
        historyIndex++;
        command = $history[$history.length - 1 - historyIndex].command;
      }
      event.preventDefault();
    } else if (event.key === 'ArrowDown') {
      if (historyIndex > -1) {
        historyIndex--;
        command = historyIndex >= 0 ? $history[$history.length - 1 - historyIndex].command : '';
      }
      event.preventDefault();
    } else if (event.key === 'Tab') {
      event.preventDefault();
      const autoCompleteCommand = Object.keys(commands).find((cmd) => cmd.startsWith(command));
      if (autoCompleteCommand) {
        command = autoCompleteCommand;
      }
    } else if (event.ctrlKey && event.key === 'l') {
      event.preventDefault();
      $history = [];
    }
  };

  const executeCommand = async () => {
      const [commandName, ...args] = command.split(' ');

      if (import.meta.env.VITE_TRACKING_ENABLED === 'true') {
        track(commandName, ...args);
      }

      const commandFunction = commands[commandName];

      if (commandFunction) {
        const output = await commandFunction(args);

        if (commandName !== 'clear') {
          if (output && typeof output === 'object' && 'type' in output && output.type === 'STREAM_START') {
            $history = [...$history, { command, outputs: [output.initialMessage] }];
            connectTextStream(output.streamKey);
          } else {
            $history = [...$history, { command, outputs: [output] }];
          }
        }
      } else {
        const output = `${commandName}: command not found`;
        $history = [...$history, { command, outputs: [output] }];
      }

      command = '';
      historyIndex = -1;
  };
</script>

<svelte:window
  on:click={() => {
    input.focus();
  }}
/>

<div class="input-wrapper">
  <input
    id="command-input"
    name="command-input"
    aria-label="Command input"
    class="hud-input"
    type="text"
    style={`color: ${$theme.foreground}`}
    bind:value={command}
    on:keydown={handleKeyDown}
    bind:this={input}
    placeholder="enter command..."
  />
</div>

<style>
  .input-wrapper {
    flex: 1;
    width: 100%;
  }

  .hud-input {
    width: 100%;
    padding: 2px 8px;
    background: transparent;
    outline: none;
    border: none;
    border-bottom: 1px solid transparent;
    caret-color: var(--hud-cyan);
    transition: border-color 0.3s ease;
    font-size: inherit;
  }

  .hud-input:focus {
    border-bottom-color: rgba(0, 240, 255, 0.3);
  }

  .hud-input::placeholder {
    color: rgba(0, 240, 255, 0.15);
    font-style: italic;
    letter-spacing: 1px;
  }
</style>
