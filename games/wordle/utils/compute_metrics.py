import re

from clemgame import get_logger


logger = get_logger(__name__)


class ComputeMetrics:
    def __init__(self):
        pass

    def extract_data_from_json(self, records_json):
        turns_data = []
        interaction = records_json["interaction"]
        for message in interaction:
            message_type = message["action"]["type"]
            if message_type == "metadata":
                error = message["action"]["content"]["error"]
                if not error:
                    turns_data.append(
                        [
                            message["action"]["content"]["guess"],
                            message["action"]["content"]["guess_feedback"],
                        ]
                    )
        return turns_data

    def num_turns(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
        """
        return len(records)

    def episodes(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
        """
        final_guess_feedback = records[-1][-1]
        final_guess_feedback = final_guess_feedback.split(" ")
        for letter in final_guess_feedback:
            if "green" not in letter:
                return 0
        return 1

    def speed(self, records, game_name):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']

        Rank is computed based on the number of turns taken to guess the word
        The lesser the number of turns, the higher the speed
        If the game is won in the first turn, speed is 100, if the game is won in the second turn, speed is 100/2, 3rd turn is 100/3 and so on
        """
        if game_name != "wordle":
            speed = 100 / len(records)
        else:
            if len(records) <= 3:
                speed = 100
            elif len(records) == 4:
                speed = 50
            elif len(records) == 5:
                speed = 30
            elif len(records) == 6:
                speed = 20
            
        return round(speed, 2)

    def change_of_opinion(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'crane', 'yes'], ['sneak', 'snake', 'no']

        Change of opinion is computed based on the number of times the opinion is changed after the critic's opinion
        """
        total_yes = 0
        total_no = 0

        use_same_guess_yes = 0
        use_diff_guess_yes = 0
        use_same_guess_no = 0
        use_diff_guess_no = 0
        overall_change = []

        for guesses in records:
            guess, guess_mod, critic_agreement = guesses
            if guess != guess_mod:
                overall_change.append(1)
                if critic_agreement == "yes":
                    total_yes += 1
                    use_diff_guess_yes += 1
                else:
                    total_no += 1
                    use_diff_guess_no += 1

            else:
                overall_change.append(0)
                if critic_agreement == "yes":
                    total_yes += 1
                    use_same_guess_yes += 1
                else:
                    total_no += 1
                    use_same_guess_no += 1

        results = {}
        results["total_yes"] = total_yes
        results["total_no"] = total_no
        results["use_same_guess_yes"] = use_same_guess_yes
        results["use_diff_guess_yes"] = use_diff_guess_yes
        results["use_same_guess_no"] = use_same_guess_no
        results["use_diff_guess_no"] = use_diff_guess_no
        results["overall_change"] = overall_change

        return results

    def turns(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
        """

        feedback_list = [record[1] for record in records]
        score_list = []

        for feedback in feedback_list:
            # Add a score of 5 for letters in green
            # Add a score of 1 for letters in yellow
            # Add a score of 0 for letters in red
            score = 0
            for letter in feedback.split(" "):
                if "green" in letter:
                    score += 5
                elif "yellow" in letter:
                    score += 3
            score_list.append(score)

        return score_list

    def turns_strategy(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'], ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
        """

        feedback = [record[1] for record in records]
        score_list = []
        if len(feedback) == 1:
            # Looks like the game was won in first guess!
            score_list = [100]
            return score_list
        # For the first turn, there is no comparison possible, hence adding the strategy score as 0
        score_list = [0]
        for guesses in zip(feedback, feedback[1:]):
            guess1, guess2 = guesses
            guess1_dict, _ = self.extract_words_by_color_code(guess1)
            _, guess2_list = self.extract_words_by_color_code(guess2)
            guess1_not_use = []
            guess1_use = []
            guess1_change = []

            if "red" in guess1_dict:
                guess1_not_use = guess1_dict["red"]
            if "green" in guess1_dict:
                guess1_use = guess1_dict["green"]
            if "yellow" in guess1_dict:
                guess1_change = guess1_dict["yellow"]
            score = 0

            result = len(set(guess1_not_use) & set(guess2_list))
            if result:
                # Decrease score by 20 for each non-used letter present in
                # next guess
                score -= result * 20

            # TODO: Do I need to penalize position change?
            result = len(set(guess1_use) & set(guess2_list))
            if result:
                # Increase score by 20 for each green letter present in next
                # guess
                score += result * 20

            # TODO: Do I need to penalize position non-change?
            result = len(set(guess1_change) & set(guess2_list))
            if result:
                # Increase score by 10 for each yellow letter present in
                # next guess
                score += result * 10
            score_list.append(score)
        return score_list

    def repeats_guess(self, records):
        """
        Assuming records contain turns_data in the below format
        [['creek', 'c<red> r<red> e<red> e<red> k<green>'],
         ['sneak', 's<green> n<yellow> e<red> a<red> k<green>']
        """
        guesses_list = [record[0] for record in records]
        repeats = int(len(guesses_list) != len(set(guesses_list)))
        num_of_repeats = len(guesses_list) - len(set(guesses_list))
        return repeats, num_of_repeats

    def extract_words_by_color_code(self, guess_word):
        position_index = 0
        color_lable_dict = {}
        letters_list = []

        for letter_code in guess_word.split(" "):
            matches = re.findall(r"\b(\w+)\b\s*\<(.+?)\>", letter_code)
            for match in matches:
                letter = match[0].strip()
                letters_list.append(letter)
                color_code = match[1].strip()
                if color_code not in color_lable_dict:
                    color_lable_dict[color_code] = []
                color_lable_dict[color_code].append(letter)
        return color_lable_dict, letters_list
