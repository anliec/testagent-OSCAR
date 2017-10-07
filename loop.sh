loop="0"

while [ $loop == "0" ]; do
	python3.6 -m pysc2.bin.agent --map CollectMineralShards --agent testagent.LearningAgent --screen_resolution 64 --norender
	result=$?
	if [ $result -ne "0" ]; then
		loop="1"
	fi
done
