### UPDATE (16.02.24): We released v0.3 of the benchmark code. The main branch will continue as v1.0-beta which has changes that effect the game code. Follow [this guide](docs/howto_update_to_v1.md) to update your game.

# clembench: A Framework for the Systematic Evaluation of Chat-Optimized Language Models as Conversational Agents

The cLLM (chat-optimized Large Language Model, "clem") framework tests such models' ability to engage in games – rule-constituted activities played using language.
The framework is a systematic way of probing for the situated language understanding of language using agents.

This repository contains the code for setting up the framework and implements a number of games that are further discussed in 

> Chalamalasetti, K., Götze, J., Hakimov, S., Madureira, B., Sadler, P., & Schlangen, D. (2023). clembench: Using Game Play to Evaluate Chat-Optimized Language Models as Conversational Agents (arXiv:2305.13455). arXiv. https://doi.org/10.48550/arXiv.2305.13455

### Evaluation Results

On the [main project website](https://clembench.github.io) , under [leaderboard](https://clembench.github.io/leaderboard.html).

### Game details

- A Simple Word Game: [taboo](docs/taboo.md)
- A Word-Guessing Game Based on Clues: [wordle](docs/wordle.md)
- Drawing Instruction Giving and Following: [image](docs/image.md)
- An ASCII Picture Reference Game: [reference](docs/reference.md)
- Scorekeeping: [private and shared](docs/privateshared.md)

## Using the benchmark

This repository is tested on `Python 3.8+`

We welcome you to contribute to or extend the benchmark with your own games and models. 
Please simply open a pull request. You can find more information on how to use the benchmark in the links below.

- [How to run the benchmark and evaluation locally](docs/howto_run_benchmark.md)
- [How to run the benchmark, update leaderboard workflow](docs/howto_benchmark_workflow.md)
- [How to add a new model as a backend](docs/howto_add_backend.md)
- [How to add and run your own game](docs/howto_add_games.md)
- [How to integrate with Slurk](docs/howto_slurk.md)
