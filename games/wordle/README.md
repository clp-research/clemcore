# A Word-Guessing Game Based on Clues: wordle

Implemented by: Kranti Chalamalasetti

The popular word guessing game ``Wordle'' gained global attention, in which players are challenged to guess a five-letter word in six attempts.  After each guess, the player receives feedback indicating which letters are in the correct position, which letters are correct but in the wrong position, and which letters are incorrect to help them strategise their next guess. The objective of the game is to guess the target word using the fewest possible guesses, and the game ends when the player guesses correctly or exhausts all six attempts.

## Wordle (Traditional Variant)
This game evaluates three key aspects of cLLM's capabilities. Firstly, it assesses how well the cLLM comprehends the game rules, which involves generating valid English words consisting of exactly five letters. Secondly, it measures how effectively cLLM uses guess feedback to generate its next guesses. Thirdly, it measures how quickly cLLM can guess the target word if it succeeds.

In traditional gameplay, cLLM plays the role of "Player A", and a deterministic wordle bot plays the role of "Player B". The game begins with the game master prompting Player A to guess the target word. The game master parses Player A's response and forwards it to Player B, which evaluates the closeness of the guess word to the target word and returns the feedback. The game master sends the feedback to Player A for the next guess and the cycle continues until the target word is guessed correctly or all six attempts are exhausted.

## Wordle (+ Semantics-Based Clue)
This is a Wordle variant, where the guesser (Player A) gets a clue before starting to guess. For example, for the target word PRIDE, the clue could be "pack of lions". The rest of the game rules follow the same as the traditional game variant. cLLM plays the role of the "player A", and a deterministic wordle bot plays the role of "player B".

The primary aim of testing this variant is to evaluate the efficacy of Player A in effectively utilising the supplementary information provided by a clue to improve its guess of the target word. The clue serves as an aid to narrow down the possible word options. The success of the game depends on Player A's ability to integrate the clue with the guess\_feedback. Player A's explanation offers insights into how the cLLM links the clue phrase and the guess\_feedback.

## Wordle (+ Clue, + Critic)
This game variant also begins with the guesser (Player A) who attempts to guess the target word based on a given clue. In contrast to other game variants, where the guessed word is immediately evaluated for its proximity to the target word, in this variant, the guessed word and the clue are forwarded to another player known as the *critic*, to get an opinion on the correctness of the guess. The critic responds with either agreement or disagreement, providing their rationale based on the information given. The critic's response is then relayed to the guesser, who can decide to stick with their initial guess or change it based on the feedback received.

This game variant helps to investigate the influence of the critic's role in the guesser's performance and can lead to interesting possibilities in human-machine interaction, where the human can be aided by the cLLM as the critic. We tested the game using the same cLLM for both roles, as well as different cLLMs for each role, employing distinct prompts for each.

## Instantiation & Evaluation

### Instantiation
In our experiments, we use a list of 2,309 possible target words and a list of 12,953 valid guess words. For textual clues, we use [New York Times crossword clues](https://www.kaggle.com/datasets/darinhawley/new-york-times-crossword-clues-answers-19932021). We sort the target words by word frequency. Out of the initial 2,309 target words, frequency details are not available for one word, and clues are not available for 39 words. These words are subsequently excluded from the experiments. The remaining 2,269 target words are sorted based on their word frequency (descending frequency) and then divided into three equal groups. The first group which contains high-frequency words, has a total of 756 words. The second group, consisting of words with medium frequency, also contains 756 words. Finally, the third group, which contains low-frequency words, has a total of 757 words. To evaluate our methodology, we chose (random seed: 42) 10 words from each frequency group, resulting in a total of 30 target words for evaluation purposes, for each game variant. As metrics, we keep track of the success rate (how often the guesser guessed the target word, within the limit of 6 guesses), the average speed (if successful, then at which turn), and for each turn closeness (based on the letter-feedback). We also keep track of whether the guesser repeats a guess (a strategic failure), and, in the critic variant, whether the guesser changes the guess after feedback from the critic.

### Error Handling
The experiments revolve closely around the cLLM models, which are expected to respond in a specific format and adhere to certain rules. However, there are multiple scenarios where the responses from these models may result in errors.

1. In the Wordle game, a subset of valid five-letter English words is used. In certain scenarios, the guesser (Player A - cLLM) may guess a valid 5-letter word that is not among the allowed guesses. In such cases, cLLM will be asked to guess another word. This reprompting process continues until cLLM makes an allowed guess.
2. The Wordle game has a strict rule that allows guessing only 5-letter words. Sometimes, the cLLM models respond with words that do not adhere to this restriction, causing the reprompting. We allow two reprompting attempts, after which the game is considered aborted.
3. Sometimes, the response of the cLLM doesn't follow the expected format as stated in the prompt. In such cases, we reprompt the cLLM to generate the response in the expected format. When faced with these circumstances, we usually give two reprompts before declaring the game as aborted.


### Evaluation
For each episode, we record the number of guesses made by the guesser. If the guesser correctly guessed the word in six or fewer attempts, the game is counted as a success. If the guesser exhausted all six attempts, the game is counted as a failure. If the guesser's response does not conform to the game rules, the game is counted as aborted. Of the successful games, the average number of guesses taken to guess the word is computed. For all the games, we also measured how close the guess gets to the target word with each turn and how many times the guesser repeats the same guess. For the episodes, where the critic is available we count how many times the guesser changed the guess based on the critic's opinion.  The following are the metrics measured for each episode.

1. **Success**: This is a binary value and measures whether the guesser guessed the target word or not.
2. **Aborted**: This is a binary value and measures whether the game aborted due to non-compliance with the game rules (words not containing 5 letters, words containing symbols other than alphabets).
3. **Speed**: This contains the score ranging from 0-to-100 and computes how quickly the guesser guessed the word.
4. **Closeness**: This contains the score ranging from 0-to-25 and determines how effectively the guesser utilizes the guess feedback. If a letter is at the correct position 5-points are awarded, and 3-points for letter at other position and 0-points for incorrect letters, leading to 25 points for a correct guess. Ideally this score should be improved across the turns.
5. **Repetition-Guesser**: This is a numeric value and assess how often the guesser repeated a guess.
6. **Change-Of-Opinion-Guesser**: This is a numeric value and calculates the number of times guesser changing/retaining the guess,
