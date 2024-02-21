from typing import Dict, List

from backends import Model
from clemgame.clemgame import Player


class Instruction:

    def __init__(self):
        self.user_messages = []
        self.system_messages = []

    def add_user_message(self, message):
        self.user_messages.append(message)

    def add_system_message(self, message):
        self.system_messages.append(message)

    def convert_to_query_messages(self):
        messages = []
        messages.append({"role": "system", "content": ""})
        for i in range(0, len(self.user_messages)):
            messages.append({"role": "user", "content": self.user_messages[i]})

            if i < len(self.system_messages):
                messages.append({"role": "assistant", "content": self.system_messages[i]})

        return messages

    def serialize(self):
        output = []

        for i in range(0, len(self.user_messages)):
            t = {"user": self.user_messages[i]}

            if i < len(self.system_messages):
                t["assistant"] = self.system_messages[i]
            output.append(t)
        return output

    def get_last_user_message(self):
        return self.user_messages[-1]

    def get_last_system_message(self):
        return self.system_messages[-1]


class InstructionFollower(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def __call__(self, instruction: Instruction, turn_idx):
        return super().__call__(instruction.convert_to_query_messages(), turn_idx)

    def _custom_response(self, messages, turn_idx):
        return "▢ P O T ▢\n▢ S ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ ▢ ▢ ▢ ▢\n▢ D A M ▢"


class InstructionGiver(Player):

    def __init__(self, model: Model):
        super().__init__(model)

    def __call__(self, instruction: Instruction, turn_idx):
        return super().__call__(instruction.convert_to_query_messages(), turn_idx)

    def _custom_response(self, messages, turn_idx):
        return "Instruction: Put X in all cells"


class ImageGame:

    def __init__(self, game_instance: Dict, player_models: List[Model]):
        self.game_id = game_instance['game_id']
        self.player_1_prompt_header = game_instance['player_1_prompt_header']
        self.player_2_prompt_header = game_instance['player_2_prompt_header']
        self.player_1_question = game_instance['player_1_question']
        self.target_grid = game_instance['target_grid']
        self.grid_dimension = game_instance['grid_dimension']
        self.number_of_letters = game_instance['number_of_letters']
        self.fill_row = game_instance['fill_row']
        self.fill_column = game_instance['fill_column']
        self.player_1_response_pattern = r'{}'.format(game_instance['player_1_response_pattern'])
        self.player_1_terminate_pattern = r'{}'.format(game_instance['player_1_terminate_pattern'])
        self.player_2_response_pattern = r'{}'.format(game_instance['player_2_response_pattern'])

        self.instruction_follower = InstructionFollower(player_models[1])
        self.instruction_giver = InstructionGiver(player_models[0])

        self.given_instruction = Instruction()
        self.given_instruction.add_user_message(
            self.player_1_prompt_header + '\n' + self.target_grid + '\n' + self.player_1_question + '\n')

        self.next_turn_message = ''
        self.followed_instruction = Instruction()

        self.current_turn = 0
        self.max_turns = self.grid_dimension * self.grid_dimension
        self.terminate = False

    def proceeds(self) -> bool:
        if self.terminate:
            return False
        return self.current_turn < self.max_turns

    def turn(self):
        # instruction giving - A side
        if self.next_turn_message != '':
            self.given_instruction.add_user_message(self.next_turn_message)
        self.next_turn_message = self.player_1_question



        player_1_prompt, player_1_response, player_1_response_text = self.instruction_giver(self.given_instruction, self.current_turn)


        # add the message to Player 1
        self.given_instruction.add_system_message(player_1_response_text)

        # reached the end on 1 side
        if 'DONE' in player_1_response_text:
            self.terminate = True
        else:

            # instruction following - 2 side
            if self.current_turn == 0:
                self.followed_instruction.add_user_message(
                    self.player_2_prompt_header + '\n' + player_1_response_text)
            else:
                self.followed_instruction.add_user_message(player_1_response_text)


            player_2_prompt, player_2_response, player_2_response_text = self.instruction_follower(self.followed_instruction, self.current_turn)


            self.followed_instruction.add_system_message(player_2_response_text)

        self.current_turn += 1
