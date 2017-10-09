"""A random agent for starcraft."""
import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization

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

class MyAgent(base_agent.BaseAgent):
    """A random agent for starcraft."""

    def step(self, obs):
        super(MyAgent, self).step(obs)
        function_id = numpy.random.choice(obs.observation["available_actions"])
        args = [[numpy.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]
        return actions.FunctionCall(function_id, args)


class MyNeuralAgent(base_agent.BaseAgent):
    """A NN agent for starcraft."""

    model = None

    def __init__(self):
        super(MyNeuralAgent, self).__init__()
        self.model = load_model("mineralshard.knn")

    def step(self, obs):
        super(MyNeuralAgent, self).step(obs)

        state = [obs.observation["screen"][features.SCREEN_FEATURES.player_relative.index],
                 obs.observation["screen"][features.SCREEN_FEATURES.selected.index]]
        formated_state = numpy.zeros(shape=(1, 64, 64, 2), dtype=float)
        for x in range(0, 63):
            for y in range(0, 63):
                formated_state[0][x][y][0] = state[0][x][y]
                formated_state[0][x][y][1] = state[1][x][y]

        action = self.model.predict(formated_state, batch_size=1)

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

