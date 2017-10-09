import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.env import sc2_env

from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


step_mul = 8

try:
    model = load_model("mineralshard.knn")
except OSError:
    sc_i = Input(shape=(64, 64, 2))
    sc_l1 = Conv2D(2, 5, activation='relu', padding='same')(sc_i)
    sc_l1n = BatchNormalization()(sc_l1)
    # reduce screen to 32x32
    sc_l2 = Conv2D(2, 5, strides=(2, 2), activation='relu', padding='same')(sc_l1n)
    sc_l2n = BatchNormalization()(sc_l2)
    sc_f = Flatten()(sc_l2n)
    d1 = Dense(256, activation='relu')(sc_f)
    oa = Dense(3)(d1)
    # output move selection at 16x16
    op = Conv2D(1, 10, strides=(2, 2), padding='same')(sc_l2n)
    # set and compile model
    model = Model(inputs=sc_i, outputs=[oa, op])
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

env = sc2_env.SC2Env(map_name="CollectMineralShards",
                     step_mul=step_mul,
                     visualize=True)

nb_actions = env.action_space.n
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)
