# Contributors

Contributors are listed alphabetically by last name.

**Anne Beyer**
Extended `GameInstanceGenerator.generate()` with optional kwargs, enabling language identifiers and multilingual game instance generation.

**Timur Berenstein**
Improved the slurk room UI (textarea input, turn-based input locking).

**Kranti Chalamalasetti**
Exposed key error classes (`GameError`, `ParseError`, `RuleViolationError`) in the public package API and contributed game master fixes.

**Sherzod Hakimov**
Major contributor to model integrations and multimodal support. Added and maintained the Cohere, Google (Gemini), Anthropic, and Together.ai backends. Extended multimodal image support across OpenAI-compatible and Google APIs. Registered numerous commercial models.

**Jonathan Jordan**
Primary developer of the HuggingFace local backend. Built and iteratively refined chat template handling, CoT support, quantized model loading, and multi-GPU inference. Also added the llama.cpp and OpenRouter backends, and contributed the largest share of open-weight model registry entries.

**Kushal Koshti**
Contributed to the early multimodal backend (later consolidated into the HuggingFace local backend) and model registry standardisation (ISO dates, open-weight flags, release dates).

**Brielen Madureira**
Contributed evaluation tooling and documentation. Built the agnostic eval script, leaderboard table formatting, and CSV export of raw scores. Authored documentation notebooks. Fixed role alternation issues in message histories.

**Shahrukh Mohiuddin**
Improved error handling in the Cohere backend.

**Filippo Momentè**
Fixed scoring bugs (interaction file filtering, model name substring collision), a vLLM backend crash, and improved padding-side detection for diverse HF model architectures.

**Karl Osswald**
Added InternVL3 multimodal model support, updated Mistral backend compatibility, and contributed HuggingFace backend improvements.

**Lara Pfennigschmidt**
Extracted `GameScorer` from `GameMaster` and implemented reprompting in `DialogueGameMaster`.

**Philipp Sadler**
Primary architect and maintainer. Responsible for the overall framework design, packaging, CLI, the registry pattern (game/model/backend/key registries with `ModelSpec` unification), the callback system for result recording, the runner architecture (sequential/batchwise), PettingZoo/Gymnasium environment wrappers, OpenEnv remote environment integration, and all major version releases.

**David Schlangen**
Added the generic OpenAI-compatible API backend and handled the OpenAI library v1 migration. Authored and maintained documentation notebooks and how-to guides.

**Paul Utsch**
Introduced the initial `EnvGameMaster` and `GameEnvironment` environment abstraction layer, which was later superseded by the PettingZoo wrappers.

**Yan Weiser**
Improved transcript utilities to include JSON-format messages.
