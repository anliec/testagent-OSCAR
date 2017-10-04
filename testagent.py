"""A random agent for starcraft."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import keras
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, Concatenate, BatchNormalization
from keras.activations import relu

_MOVE_ACTION = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOP = actions.FUNCTIONS.no_op.id


class MyAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def step(self, obs):
        super(MyAgent, self).step(obs)
        function_id = numpy.random.choice(obs.observation["available_actions"])
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)


class MyNeuralAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    model = None

    def __init__(self):
        super(MyNeuralAgent, self).__init__()
        try:
            self.model = load_model("mineralshard.knn")
        except OSError:
            pr_i = Input(shape=(64, 64, 1))
            pr_l1 = Conv2D(1, 5, activation='sigmoid', padding='same')(pr_i)
            pr_l2 = Conv2D(1, 3, activation='sigmoid', padding='same')(pr_l1)
            # pr_n = BatchNormalization()(pr_l2)
            pr_f = Flatten()(pr_l2)
            sl_i = Input(shape=(64, 64, 1))
            sl_l1 = Conv2D(1, 5, activation='sigmoid', padding='same')(sl_i)
            sl_l2 = Conv2D(1, 3, activation='sigmoid', padding='same')(sl_l1)
            # sl_n = BatchNormalization()(sl_l2)
            sl_f = Flatten()(sl_l2)
            mrg = Concatenate(axis=-1)([pr_f, sl_f])
            d1 = Dense(256, activation='sigmoid')(mrg)
            # d1_n = BatchNormalization()(d1)
            oa = Dense(3, activation="sigmoid")(d1)
            op = Conv2D(1, 10, activation='sigmoid', padding='same')(pr_l2)
            self.model = Model(inputs=[pr_i, sl_i], outputs=[oa, op])
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['mean_squared_error'])

    def step(self, obs):
        super(MyNeuralAgent, self).step(obs)

        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        pr_formated_state = numpy.zeros(shape=(1, 64, 64, 1), dtype=float)
        sl_formated_state = numpy.zeros(shape=(1, 64, 64, 1), dtype=float)
        for x in range(0, 63):
            for y in range(0, 63):
                pr_formated_state[0][x][y][0] = state[0][x][y]
                sl_formated_state[0][x][y][0] = state[1][x][y]
        action = self.model.predict([pr_formated_state, sl_formated_state], batch_size=1)

        action_vector = action[0][0]
        # print(action_vector)
        # print(action_vector.shape)
        # remove not playable action
        if _MOVE_ACTION not in obs.observation["available_actions"]:
            action_vector[1] = 0.0
        if _SELECT_ARMY not in obs.observation["available_actions"]:
            action_vector[2] = 0.0
        # select best score
        best_action_id = 0
        best_score = 0.0
        for i in range(0, len(action_vector) - 1):
            if best_score < action_vector[i]:
                best_score = action_vector[i]
                best_action_id = i

        if best_action_id == 1:
            selected_action = _MOVE_ACTION
        elif best_action_id == 2:
            selected_action = _SELECT_ARMY
        else:
            selected_action = _NOP

        action_args = []
        # print(action[1][0].shape)
        if best_action_id == 1:
            position_vector = action[1][0]
            max_x = 0
            max_y = 0
            max_val = -999.9
            for x in range(0, 63):
                for y in range(0, 63):
                    if position_vector[x][y][0] >= max_val:
                        max_val = position_vector[x][y][0]
                        max_x = x
                        max_y = y
            print("target: " + str(max_x) + ":" + str(max_y) + " as value is " + str(max_val))
            action_args = [[0], [max_x, max_y]]
        elif best_action_id == 2:
            # select all
            action_args = [[0]]

        if _MOVE_SCREEN not in obs.observation["available_actions"] and best_action_id != 2:
            print("manual selection !")
            return actions.FunctionCall(_SELECT_ARMY, [[0]])
        return actions.FunctionCall(selected_action, action_args)


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
    """A random agent for starcraft."""

    model = None
    learning_batch_input = []
    learning_batch_output = []

    def __init__(self):
        super(LearningAgent, self).__init__()
        try:
            self.model = load_model("mineralshard.knn")
        except OSError:
            pr_i = Input(shape=(64, 64, 1))
            pr_in = BatchNormalization()(pr_i)
            pr_l1 = Conv2D(1, 5, activation='sigmoid', padding='same')(pr_i)
            pr_l2 = Conv2D(1, 3, activation='sigmoid', padding='same')(pr_l1)
            # pr_n = BatchNormalization()(pr_l2)
            pr_f = Flatten()(pr_l2)
            sl_i = Input(shape=(64, 64, 1))
            sl_in = BatchNormalization()(sl_i)
            sl_l1 = Conv2D(1, 5, activation='sigmoid', padding='same')(sl_i)
            sl_l2 = Conv2D(1, 3, activation='sigmoid', padding='same')(sl_l1)
            # sl_n = BatchNormalization()(sl_l2)
            sl_f = Flatten()(sl_l2)
            mrg = Concatenate(axis=-1)([pr_f, sl_f])
            d1 = Dense(256, activation='sigmoid')(mrg)
            # d1_n = BatchNormalization()(d1)
            oa = Dense(3, activation="sigmoid")(d1)
            op = Conv2D(1, 5, activation='sigmoid', padding='same')(pr_l2)
            self.model = Model(inputs=[pr_i, sl_i], outputs=[oa, op])
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['mean_squared_error'])

    def step(self, obs):
        selected_action = None
        selected_arg = None
        super(LearningAgent, self).step(obs)
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            if not neutral_y.any() or not player_y.any():
                selected_action = _NO_OP
                selected_arg = []
                return actions.FunctionCall(selected_action, selected_arg)
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
        position_array = numpy.zeros(shape=(64, 64, 1), dtype=float)
        player_rel = obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index]
        for x in range(0, 63):
            for y in range(0, 63):
                position_array[x][y][0] = player_rel[x][y] / 6.0
        # numpy.copyto(position_array, obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index])
        if selected_action == _NO_OP:
            action_array[0] = 1.0
        elif selected_action == _MOVE_SCREEN:
            action_array[1] = 1.0
            position_array[selected_arg[1][0]][selected_arg[1][1]][0] = 1.0
        elif selected_action == _SELECT_ARMY:
            action_array[2] = 1.0
        # convert state to NN format
        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        # if numpy.isnan(numpy.sum(state[0])):
        #     print("NaN on player_relative !")
        # if numpy.isnan(numpy.sum(state[1])):
        #     print("NaN on selected !")
        pr_formated_state = numpy.zeros(shape=(64, 64, 1), dtype=float)
        sl_formated_state = numpy.zeros(shape=(64, 64, 1), dtype=float)
        for x in range(0, 63):
            for y in range(0, 63):
                pr_formated_state[x][y][0] = state[0][x][y]
                sl_formated_state[x][y][0] = state[1][x][y]
        inputs = [pr_formated_state, sl_formated_state]
        outputs = [action_array, position_array]
        self.learning_batch_input.append(inputs)
        self.learning_batch_output.append(outputs)
        # if numpy.isnan(numpy.sum(pr_formated_state)):
        #     print("NaN on pr_formated_state !")
        # if numpy.isnan(numpy.sum(sl_formated_state)):
        #     print("NaN on sl_formated_state !")
        # if numpy.isnan(numpy.sum(action_array)):
        #     print("NaN on action_array !")
        # if numpy.isnan(numpy.sum(position_array)):
        #     print("NaN on position_array !")
        return actions.FunctionCall(selected_action, selected_arg)

    def reset(self):
        super(LearningAgent, self).reset()
        if len(self.learning_batch_output) == len(self.learning_batch_input) and len(self.learning_batch_input) != 0:
            # print("list of len " + str(len(self.learning_batch_input)) + " and of shape " +
            #       str(self.learning_batch_input[0][0].shape))
            # print("list of len " + str(len(self.learning_batch_output)) + " and of shape [" +
            #       str(self.learning_batch_output[0][0].shape) + ", " +
            #       str(self.learning_batch_output[0][1].shape))
            batch_size = len(self.learning_batch_output)
            pr_array = numpy.zeros(shape=(batch_size, 64, 64, 1), dtype=float)
            sl_array = numpy.zeros(shape=(batch_size, 64, 64, 1), dtype=float)
            i = 0
            for it in self.learning_batch_input:
                pr_array[i] = it[0]
                sl_array[i] = it[1]
                i += 1
            action_array = numpy.zeros(shape=(batch_size, 3), dtype=float)
            pos_array = numpy.zeros(shape=(batch_size, 64, 64, 1), dtype=float)
            i = 0
            for it in self.learning_batch_output:
                action_array[i] = it[0]
                pos_array[i] = it[1]
                i += 1
            # if numpy.isnan(numpy.sum(pr_array)):
            #     print("NaN on pr_array !")
            # if numpy.isnan(numpy.sum(sl_array)):
            #     print("NaN on sl_array !")
            # if numpy.isnan(numpy.sum(action_array)):
            #     print("NaN on action_array !")
            # if numpy.isnan(numpy.sum(pos_array)):
            #     print("NaN on pos_array !")
            # print("pr_array: [" + str(numpy.min(pr_array)) + ";" + str(numpy.max(pr_array)) + "]")
            # print("sl_array: [" + str(numpy.min(sl_array)) + ";" + str(numpy.max(sl_array)) + "]")
            # print("action_array: [" + str(numpy.min(action_array)) + ";" + str(numpy.max(action_array)) + "]")
            # print("pos_array: [" + str(numpy.min(pos_array)) + ";" + str(numpy.max(pos_array)) + "]")
            self.model.fit(x=[pr_array, sl_array],
                           y=[action_array, pos_array],
                           batch_size=batch_size,
                           nb_epoch=5,
                           verbose=1,
                           validation_split=0.2)
            self.learning_batch_output = []
            self.learning_batch_input = []
            save_model(self.model, "mineralshard.knn")
