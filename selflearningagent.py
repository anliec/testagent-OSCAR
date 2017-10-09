import numpy
import sys
import math
import scipy.misc
import time

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization

from neuralmodel import get_neural_network

_MOVE_ACTION = actions.FUNCTIONS.Move_screen.id
_NOP = actions.FUNCTIONS.no_op.id

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

class SelfLearningAgent(base_agent.BaseAgent):
    """A SelfLearning agent for learning starcraft."""

    def __init__(self):
        super(SelfLearningAgent, self).__init__()
        self.model = get_neural_network()

    def step(self, obs):
        super(SelfLearningAgent, self).step(obs)

        # set action target
        # action_array = numpy.array([0.0, 0.0, 0.0])
        action_array = numpy.zeros(shape=(1, 3), dtype=float)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            # if it is possible to move, move
            action_array[0][1] = 1.0
        else:
            # else select army
            action_array[0][2] = 1.0

        # set position target
        position_array = numpy.zeros(shape=(1, 16, 16, 1), dtype=float)
        # but set it to non zero value only if a move is wanted
        if action_array[0][1] == 1.0:
            # find position of the two marines
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
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
                        distance = 91 # > sqrt(64²+64²) ~ 90.5
                        for marine in zip(player_y, player_x):
                            distance = min(numpy.linalg.norm(numpy.array([x, y]) - numpy.array(marine)), distance)
                        # if not min_distance or distance < min_distance:
                        #     min_distance = distance
                        #     min_x = pos_x
                        #     min_y = pos_y
                        # score = 1.0 - distance / 100.0
                        score = max(1.0 - math.sqrt(distance / 30), 0.2)
                        position_array[0][pos_x][pos_y][0] = max(score, position_array[0][pos_x][pos_y][0])
                        # im_array[pos_x][pos_y] = position_array[0][pos_x][pos_y][0] * 255.0
        # convert state to NN format
        state = [obs.observation["screen"][_PLAYER_RELATIVE],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formatted_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
        # for x in range(0, 63):
        #     for y in range(0, 63):
        #         formatted_state[0][x][y][0] = state[0][x][y]
        #         formatted_state[0][x][y][1] = state[1][x][y]
        for formatted_row, state0_row, state1_row in zip(formatted_state[0], state[0], state[1]):
            for formatted_case, state0_case, state1_case in zip(formatted_row, state0_row, state1_row):
                formatted_case[0] = state0_case
                formatted_case[1] = state1_case

        # for x, y in zip(player_y, player_x):
        #     pos_x = min(round(int(x) / 4), 15)
        #     pos_y = min(round(int(y) / 4), 15)
        #     im_array[pos_x][pos_y] = 255
        # im = scipy.misc.toimage(im_array, high=255.0, low=0.0)
        # im = scipy.misc.imresize(im, (256, 256), interp='nearest')
        # scipy.misc.imsave('outfile.jpg', im)
        # time.sleep(1)

        self.model.fit(x=formatted_state,
                       y=[action_array, position_array],
                       batch_size=1,
                       nb_epoch=1,
                       verbose=0,
                       validation_split=0)

        # use NN to select an action
        action = self.model.predict(formatted_state, batch_size=1)
        # action = [action_array, position_array]
        # play that action
        action_vector = action[0][0]
        # remove not playable action
        if _MOVE_SCREEN not in obs.observation["available_actions"]:
            action_vector[1] = -1.0
        if _SELECT_ARMY not in obs.observation["available_actions"]:
            action_vector[2] = -1.0
        # select best score
        best_action_id = numpy.argmax(action_vector)

        if best_action_id == 1:
            selected_action = _MOVE_ACTION
        elif best_action_id == 2:
            selected_action = _SELECT_ARMY
            # print("NN choose selection !")
        else:
            selected_action = _NOP
            print("NN choose to do nothing")

        action_args = []
        if best_action_id == 1:
            position_vector = action[1][0]
            # get the best position according to "score"
            max_coordinate = numpy.argmax(position_vector)
            x = (max_coordinate % 16) * 4
            y = (max_coordinate // 16) * 4
            action_args = [[0], [x, y]]
        elif best_action_id == 2:
            # select all
            action_args = [[0]]

        return actions.FunctionCall(selected_action, action_args)

    def reset(self):
        super(SelfLearningAgent, self).reset()
        save_model(self.model, "mineralshard.knn")
