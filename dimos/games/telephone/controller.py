"""GameController: orchestrates rounds of Robot Telephone."""

import json
import time
from dataclasses import dataclass, field
from threading import Thread
from typing import Any

from langchain.chat_models import init_chat_model

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.utils.logging_config import setup_logger
from reactivex.disposable import Disposable

logger = setup_logger()


@dataclass
class RoundTranscript:
    round_number: int = 0
    target_object: str = ""
    a_description: str = ""
    b_relay: str = ""
    c_answer: str = ""
    score: dict[str, Any] = field(default_factory=dict)


class GameControllerConfig(ModuleConfig):
    judge_model: str = "gpt-4o"


class GameController(Module[GameControllerConfig]):
    """Orchestrates Robot Telephone game rounds."""

    default_config = GameControllerConfig

    game_command: Out[str]
    search_result: In[str]
    transcript_a: In[str]
    transcript_b: In[str]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._round = 0
        self._current: RoundTranscript = RoundTranscript()
        self._history: list[RoundTranscript] = []

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.transcript_a.subscribe(self._on_a)))
        self._disposables.add(Disposable(self.transcript_b.subscribe(self._on_b)))
        self._disposables.add(Disposable(self.search_result.subscribe(self._on_result)))

    def _on_a(self, description: str) -> None:
        self._current.a_description = description

    def _on_b(self, relay: str) -> None:
        self._current.b_relay = relay

    def _on_result(self, result: str) -> None:
        self._current.c_answer = result
        thread = Thread(target=self._score_round, daemon=True)
        thread.start()

    @skill
    def start_round(self, target_object: str = "the most interesting object") -> str:
        """Start a new Robot Telephone round.

        Args:
            target_object: Hint for what Robot A should focus on (optional).

        Returns:
            str: Status message.
        """
        self._round += 1
        self._current = RoundTranscript(
            round_number=self._round,
            target_object=target_object,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  ROUND {self._round}  |  Target hint: {target_object}")
        logger.info(f"{'=' * 60}\n")

        self.game_command.publish("DESCRIBE")

        return f"Round {self._round} started. Robot A is describing..."

    def _score_round(self) -> None:
        try:
            judge = init_chat_model(self.config.judge_model)

            prompt = (
                f"You are judging a game of Robot Telephone.\n\n"
                f"Target hint: {self._current.target_object}\n"
                f'Robot A described: "{self._current.a_description}"\n'
                f'Robot B relayed as: "{self._current.b_relay}"\n'
                f'Robot C concluded: "{self._current.c_answer}"\n\n'
                f"Score each (0-10):\n"
                f"- a_accuracy: How well did A describe what it saw?\n"
                f"- b_fidelity: How faithful was B's relay to A's description?\n"
                f"- c_success: Did C correctly identify the object?\n"
                f"- telephone_drift: How much did the message change? (10=no drift, 0=completely lost)\n"
                f"- overall: Overall game score\n\n"
                f'Respond as JSON only: {{"a_accuracy": N, "b_fidelity": N, '
                f'"c_success": N, "telephone_drift": N, "overall": N, "commentary": "..."}}'
            )

            response = judge.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)

            # Strip markdown code fences if present (```json ... ```)
            text = content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            try:
                score = json.loads(text)
            except json.JSONDecodeError:
                score = {"raw_response": content}

            self._current.score = score
            self._history.append(self._current)

            logger.info(f"\n{'=' * 60}")
            logger.info(f"  ROUND {self._current.round_number} RESULTS")
            logger.info(f"{'=' * 60}")
            logger.info(f'  [A] "{self._current.a_description}"')
            logger.info(f'  [B] "{self._current.b_relay}"')
            logger.info(f'  [C] "{self._current.c_answer}"')
            logger.info(f"  Score: {score}")
            logger.info(f"{'=' * 60}\n")

        except Exception as e:
            logger.error(f"Scoring failed: {e}")

    @rpc
    def get_history(self) -> list[dict[str, Any]]:
        """Get all round transcripts and scores."""
        return [
            {
                "round": t.round_number,
                "target": t.target_object,
                "a": t.a_description,
                "b": t.b_relay,
                "c": t.c_answer,
                "score": t.score,
            }
            for t in self._history
        ]


game_controller = GameController.blueprint

__all__ = ["GameController", "game_controller"]
