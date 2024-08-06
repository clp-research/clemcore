# A Simple Word Game: taboo

Implemented by: Jana Götze & Philipp Sadler

In this game one player (the Describer) describes a target word for another player to guess (Guesser).
The Describer must explain the target word concept without using neither the word itself, nor a part of the word.
For example, when the target word is _flashlight_, the Describer cannot use the words _light_ or _flash_.
After each incorrect guess by the Guesser, the Describer can add to his description.
The game ends when the Guesser guesses correctly or a maximum number of turns has been reached.

The game tests a cLLM's ability to describe concepts and give meaning definitions.
It also tests whether it can detect parts of words and its helpfulness in the game context.
For example, if a Describer does not alter or extend its initial description after an incorrect guess, we consider this unhelpful behavior.
On the opposite side, if a Guesser repeats an earlier guess, it has not understood the game goal well enough to make real progress at each turn.


### Instantiation
We instantiate this game by setting the maximum number of guesses to 3 and we use target words that vary according to their frequency.

We use an [English word frequency list based on web data](https://www.kaggle.com/datasets/rtatman/english-word-frequency) to derive a list of lemmatized content words from which we choose words from 3 levels of frequency by dividing the word list into 3 equally-sized bins after removing words with a frequency of less than 5 per 1 million tokens.
The remaining low-frequency words occur up $9.4$ times per 1 million tokens, the medium-frequency words occur up to $25.1$ times per 1 million tokens and the high-frequency tokens occur up to $12951.3$ times in 1 million tokens.
After a random selection from each frequency level, we manually ensure that the final word list does not contain inappropriate words such as vulgar language or proper names.
The final word lists contain 10 words each.
We use word frequency as a proxy for the difficulty of a game instance.

### Evaluation

We measure the following metrics at the episode-level:

1. **Success**: Whether or not the Guesser guessed the target word.
2. **Abort**: 1 if any player did not follow the rules, and 0 otherwise.
3. **Speed**: How early the Guesser guessed the word as measured by 100/t, where t is the turn number in which the target was found. When the game was unsuccessful or aborted speed is undefined.
4. **Repetition-Guesser**: How often the Guesser repeated a guess, as the maximum number that the Guesser repeated a word in consecutive turns.
5. **Repetition-Describer**: How often the Describer repeated itself, as the maximum number of repetitions in consecutive turns.
