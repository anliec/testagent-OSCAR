#!/bin/bash

#python3.6 -m pysc2.bin.agent --map CollectMineralShards --agent testagent.LearningAgent --screen_resolution 64 --norender
python3.6 -m pysc2.bin.agent --map CollectMineralShards --agent testagent.MyNeuralAgent --screen_resolution 64
