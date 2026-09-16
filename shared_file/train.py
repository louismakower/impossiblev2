"""RL on the shared-file environment.

One policy plays both agents. Each episode starts a fresh container with an
empty shared file and runs the two agents in it concurrently, each with bash
and the append tool, until both have submitted or run out of turns. Each
agent's reward for the episode comes from `reward.py`, applied to the final
contents of the shared file. Every turn either agent took is a training
example for the one policy, weighted by that agent's advantage.

Usage:
    python train.py --steps 200 --episodes 16
"""

import argparse
import statistics

import tinker
from tinker import types

# Runs both agents in a container and returns the episode: `lines` is the
# final contents of the shared file, `turns[agent]` the agent's assistant
# turns, each with `prompt_tokens`, `sampled_tokens` and `logprobs`.
from harness import run_episode
from reward import AGENTS, count_lines, in_bonus_range, rewards

BASE_MODEL = "moonshotai/Kimi-K2.6"
LORA_RANK = 32
LEARNING_RATE = 1e-5
MAX_TOKENS = 4096


def turn_datum(turn, advantage: float) -> types.Datum:
    """One assistant turn as an importance-sampling example.

    The model input is the prompt followed by the sampled tokens, shifted by
    one against the targets. Prompt positions get zero advantage so only the
    sampled tokens carry gradient.
    """
    tokens = turn.prompt_tokens + turn.sampled_tokens
    prompt = len(turn.prompt_tokens) - 1
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "logprobs": [0.0] * prompt + turn.logprobs,
            "advantages": [0.0] * prompt + [advantage] * len(turn.sampled_tokens),
        },
    )


def train(steps: int, episodes: int) -> None:
    service = tinker.ServiceClient()
    training = service.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)

    for step in range(steps):
        sampler = training.save_weights_and_get_sampling_client(name=f"step-{step:04d}")
        results = [run_episode(sampler, AGENTS, max_tokens=MAX_TOKENS) for _ in range(episodes)]
        counts = [count_lines(r.lines) for r in results]
        episode_rewards = [rewards(c) for c in counts]

        # One baseline over both agents' rewards: they are the same policy.
        baseline = statistics.mean(rw[a] for rw in episode_rewards for a in AGENTS)
        data = [
            turn_datum(turn, rw[agent] - baseline)
            for result, rw in zip(results, episode_rewards)
            for agent in AGENTS
            for turn in result.turns[agent]
        ]
        training.forward_backward(data, loss_fn="importance_sampling").result()
        training.optim_step(types.AdamParams(learning_rate=LEARNING_RATE)).result()

        bonus_rate = statistics.mean(in_bonus_range(c) for c in counts)
        print(f"step {step}: mean reward {baseline:.3f}, bonus rate {bonus_rate:.2f}, {len(data)} turns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=16, help="episodes per optimizer step")
    args = parser.parse_args()
    train(args.steps, args.episodes)
