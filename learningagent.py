import numpy
import math

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization

from neuralmodel import get_neural_network

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]


class LearningAgent(base_agent.BaseAgent):
    """A scripted agent for learning starcraft."""

    def __init__(self):
        super(LearningAgent, self).__init__()
        self.learning_batch_input = []
        self.learning_batch_output = [[], []]
        self.model = get_neural_network()

    def step(self, obs):
        super(LearningAgent, self).step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            if not neutral_y.any() or not player_y.any():
                selected_action = _NO_OP
                selected_arg = []
            else:
                player = [int(player_x.mean()), int(player_y.mean())]
                closest, min_dist = None, None
                for p in zip(neutral_x, neutral_y):
                    dist = numpy.linalg.norm(numpy.array(player) - numpy.array(p))
                    if not min_dist or dist < min_dist:
                        closest, min_dist = p, dist
                selected_action = _MOVE_SCREEN
                selected_arg = [_NOT_QUEUED, closest]
        else:
            selected_action = _SELECT_ARMY
            selected_arg = [_SELECT_ALL]
        action_array = numpy.array([0.0, 0.0, 0.0])
        if selected_action == _NO_OP:
            action_array[0] = 1.0
        elif selected_action == _MOVE_SCREEN:
            action_array[1] = 1.0
        elif selected_action == _SELECT_ARMY:
            action_array[2] = 1.0
        # set pos target
        # get current screen
        player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
        # find position of the two marines
        player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
        # neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
        # marines = [int(player_x.mean()), int(player_y.mean())]
        # marines = [(player_x[0], player_y[0])]
        # for x, y in zip(player_x, player_y):
        #     close_to_marines = False
        #     for mx, my in marines:
        #         if abs(x - mx) < 1 and abs(y - my) < 1:
        #             close_to_marines = True
        #             break
        #     if not close_to_marines:
        #     marines.append((x, y))
        # set a target weight on move action choose
        position_array = numpy.zeros(shape=(16, 16, 1), dtype=float)
        # min_distance, min_x, min_y = None, None, None
        # for x, y in zip(neutral_x, neutral_y):
        #     pos_x = min(round(int(x) / 4), 15)
        #     pos_y = min(round(int(y) / 4), 15)
        #     distance = 91  # > sqrt(64²+64²) ~ 90.5
        #     for marine in zip(player_x, player_y):
        #         distance = min(numpy.linalg.norm(numpy.array([x, y]) - numpy.array(marine)), distance)
        #     # if not min_distance or distance < min_distance:
        #     #     min_distance = distance
        #     #     min_x = pos_x
        #     #     min_y = pos_y
        #     # score = 1.0 - distance / 100.0
        #     score = 1.0 - math.sqrt(distance / 91)
        #     position_array[0][pos_x][pos_y][0] = max(score, position_array[0][pos_x][pos_y][0])
        # im_array = numpy.zeros(shape=(16, 16))
        for x, row in enumerate(player_relative):
            for y, case in enumerate(row):
                if case == 3:
                    pos_x = min(round(x / 4), 15)
                    pos_y = min(round(y / 4), 15)
                    distance = 91  # > sqrt(64²+64²) ~ 90.5
                    for marine in zip(player_y, player_x):
                        distance = min(numpy.linalg.norm(numpy.array([x, y]) - numpy.array(marine)), distance)
                    # if not min_distance or distance < min_distance:
                    #     min_distance = distance
                    #     min_x = pos_x
                    #     min_y = pos_y
                    # score = 1.0 - distance / 100.0
                    score = max(1.0 - math.sqrt(distance / 30), 0.2)
                    position_array[pos_x][pos_y][0] = max(score, position_array[pos_x][pos_y][0])
                    # im_array[pos_x][pos_y] = position_array[0][pos_x][pos_y][0] * 255.0
        # convert state to NN format
        state = [player_relative,
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formatted_state = numpy.zeros(shape=(64, 64, 2), dtype=float)
        # for x in range(0, 63):
        #     for y in range(0, 63):
        #         formatted_state[0][x][y][0] = state[0][x][y]
        #         formatted_state[0][x][y][1] = state[1][x][y]
        for formatted_row, state0_row, state1_row in zip(formatted_state, state[0], state[1]):
            for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                formatted_case[0] = state0_case
                formatted_case[1] = state1_case
        # add current state and target to learning list
        self.learning_batch_input.append(formatted_state)
        self.learning_batch_output[0].append(action_array)
        self.learning_batch_output[1].append(position_array)
        return actions.FunctionCall(selected_action, selected_arg)

    def reset(self):
        super(LearningAgent, self).reset()
        if len(self.learning_batch_input) != 0:
            array = numpy.array(self.learning_batch_input)
            action_array = numpy.array(self.learning_batch_output[0])
            pos_array = numpy.array(self.learning_batch_output[1], dtype=int)
            self.model.fit(x=array,
                           y=[action_array, pos_array],
                           batch_size=1,
                           nb_epoch=1,
                           verbose=1,
                           validation_split=0)
            self.learning_batch_output = [[], []]
            self.learning_batch_input = []
            save_model(self.model, "mineralshard.knn")



