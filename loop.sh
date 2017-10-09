loop="0"

while [ $loop == "0" ]; do
	./launcher_selflearn.sh
	result=$?
	if [ $result -ne "0" ]; then
		loop="1"
	fi
done
