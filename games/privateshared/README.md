# Scorekeeping: private and shared

Implemented by: Brielen Madureira

In an interaction, one device of the *conversational grounding* anchoring process is that participants coordinate what is private knowledge and what information has already been shared in previous turns. After each utterance, the status of novel information should be updated from private to shared in both agents' discourse models. This is how they do *scorekeeping*, i.e. keeping track of the *common ground* which is built incrementally, turn by turn.

For example, consider a slot-filling conversation with asymmetric roles between a questioner and an answerer, which can occur as part of customer service, job interviews or medical diagnosis interactions. If the questioner asks *Where do you work?*, at this point this is typically private information that only the answerer knows. After the reply, the place of work becomes shared information, and both the questioner and the answerer know that.

The evaluation method for scorekeeping proposed by \citet{madureira-schlangen-2022-visual} is to probe, after each turn, whether the dialogue model's representations correctly encode information about the private or shared status of true and false statements. With cLLMs, we can instead probe by directly posing side questions to an agent while it interacts with another agent.

We thus introduce a dialogue game which enables testing the scorekeeping abilities of these models, by measuring how well the cLLM's discourse model gets correctly updated after each turn.

### Game Description
This is a slot-filling conversation, mediated by a game master, with asymmetric roles between a questioner and an answerer. We define $n$ slots to be filled. The answerer player $A$ privately knows the values of all slots from the beginning of the interaction (passed via an initial prompt) but the questioner $Q$ does not. The questioner then asks $n$ questions, %$(q_i)_{i=1}^n$,
one by one, aiming at filling those slots based on $A$'s answers. %$(a_i)_{i=1}^n$.
A final state is reached when $Q$ fills all the slots and the the goal state is having all values correctly filled.

Before the interaction starts and after each question-answer pair, %$(q_i, a_i)$,
the game master probes the agent's discourse model by asking about the status (private or shared) of every slot, one by one and in a random order, in the conversation so far. This results in a sequence of $n+1$ probing rounds, each containing $n$ binary decisions, which can be used to evaluate the performance of the model.% in this dialogue game.

### Instantiation
Here we introduce two versions of this setting with $5$ slots: i) a travel agent and a customer booking a trip and ii) a recruiter and a job applicant in a job interview. We implement the questioner programmatically and let the cLLM play the role of the answerer. This game is an example of a "messenger" setup, where the game master plays a more active role, by parsing responses and performing the probing rounds. The game master begins by instructing the cLLM about the setting, explaining that it should give replies according to the given values for each slot.

Besides the task-oriented requests from $Q$, the cLLM must also respond to probing questions privately posed by the game master. The initial prompt defines special labels to be used for each type of question and response. Because the questioner's order of requests is under the control of the game master, the truth values are known and can be immediately compared to the cLLM' answers. For completeness, we also make the probing before any move from the questioner.  Note that, in the first probing round, all slot values are private, whereas in the last one, all are shared.

**(i) Travel Agency**: simulates a conversation between a customer (the cLLM) and a travel agent. The customer wishes to book a trip according to a set of 5 slots: `from` (origin), `to` (destination), `by` (means of transportation), `class` and `when` (time of departure). For probing, the game master can ask, for instance, *"Does the travel agent know where you want to go?"*. The correct answer is *no* until the travel agent has received a reply for that slot, when the correct answer changes to *yes*.

**(i) Job Interview**: simulates a conversation between a job applicant (the cLLM) and a recruiter. The job applicant has a CV with 5 slots: `bachelor`, `industry experience`, `highest education`, `other skills` and `availability`. For probing, the game master can ask, for instance, *"Has the recruiter been informed about your availability?"*. Again, the correct answer is *no* until the recruiter has received a reply for that slot.

### Implementation
For each version, we generate 10 instances by randomly selecting values for all slots and a random order for the questioner's requests. The cLLM is prompted to only give short, direct answers to avoid that slot values are given in anticipation.
We consider that a slot was filled if the answer contains its value. We also check whether it contains any new value and update the probing ground truth accordingly. In probing rounds, the game master prompts the model to answer yes or no. If, for some reason, it was not possible to parse a valid response during probing, we add additional instructions for clarity in the request. After the maximum number of $5$ failed attempts, an invalid response symbol is used instead and the game will be aborted.
Each probing question is posed on its own and does not get appended to the dialogue context in subsequent turns. For instance, after $(q_i, a_i)$, the $i+1$-th sequence of probes is made. At request $i+1$, however, the dialogue context contains only the game questions and answers up to turn $i$ and none of the probes. If the agent does not follow the game rules (i.e. does not use the correct tags according to the instructions or give answers that cannot be parsed in probing), the game is aborted.

### Evaluation
Besides following the game instructions, a competent player should i) provide the correct slot value to answer each question accordingly; and ii) know, at any point, which slot values have already been disclosed to the questioner and what has not yet been revealed. Specifically, the exact turn when a slot value shifts from private to shared should be correctly detected.

**Turn-Level Scores**: At each turn, the game master collects $n$ binary answers (yes or no). We thus use **accuracy** as a turn-level score, computed by comparing these $n$ answers to the corresponding $n$ truth values. An ideal model would achieve high accuracy at all turns. We also track a binary label which is 1 if the current slot is correctly given in the answer.

**Episode-Level Scores**: At the end of an episode, $(n+1)n$ answers have been collected via probing. We compute **accuracy** across all answers. However, given that this is a binary classification task, the random performance is very high. We thus also compute **Cohen's kappa** \citep{cohen-kappa} as an episode-level score, truncated at 0.

As discussed in \cite{madureira-schlangen-2022-visual}, a model biased towards considering all values as private would perform well at initial turns, whereas models biased towards shared would perform well at final turns. We follow their suggestion to also evaluate the performance in middle turns, where the distribution of labels is more balanced. For that, we report the accuracy at the third probing round, namely **middle-accuracy**.

The validity of the results rely on the slots having been correctly filled. As a sanity check, we compute the proportion of answers that contain the correct slot value as an additional episode level score (**slot-filling-accuracy**). (However, even if the cLLM hallucinates an answer, the probing can still be performed, because a wrong value is still a shared value.)

**Preferred Score**: The harmonic mean between slot-filling-accuracy and truncated $\kappa$ is normalised to $[0, 100$ and used as the main score, summarising the performance of an agent in an episode.

# Important implementation details

- Slot filling is tested by checking if the answer contains a string. For that to be evaluated correctly, no overlap is allowed in slot values (i.e. a slot value must not be contained in another value). If you add other experiments/domains, run ```games/privateshaed/checkvalues.py``` to check if everything is fine.
- If you want to generate new instances, change the ```SEED``` constant in ```instancegenerator.py```.
