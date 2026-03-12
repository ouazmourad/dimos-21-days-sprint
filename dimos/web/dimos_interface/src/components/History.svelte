<script lang="ts">
  import { history } from '../stores/history';
  import { theme } from '../stores/theme';
  import Ps1 from './Ps1.svelte';
</script>

{#each $history as { command, outputs }, i}
  <div class="history-entry" style={`color: ${$theme.foreground}`}>
    <div class="command-line">
      <Ps1 />
      <span class="command-text">{command}</span>
    </div>

    {#each outputs as output}
      <p class="output-text">
        {output}
      </p>
    {/each}
  </div>
{/each}

<style>
  .history-entry {
    margin-bottom: 4px;
    padding-left: 2px;
    border-left: 1px solid transparent;
    transition: border-color 0.2s;
  }

  .history-entry:hover {
    border-left-color: rgba(0, 240, 255, 0.2);
  }

  .command-line {
    display: flex;
    flex-direction: row;
    align-items: center;
  }

  @media (max-width: 768px) {
    .command-line {
      flex-direction: column;
      align-items: flex-start;
    }
  }

  .command-text {
    padding-left: 8px;
    opacity: 0.9;
  }

  .output-text {
    white-space: pre;
    padding-left: 4px;
    opacity: 0.7;
    line-height: 1.4;
  }
</style>
