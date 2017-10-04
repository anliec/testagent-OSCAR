while [ 0 == 0 ]; do
	python3.6 -m pysc2.bin.agent --map CollectMineralShards --agent testagent.LearningAgent --screen_resolution 64 --norender
done
