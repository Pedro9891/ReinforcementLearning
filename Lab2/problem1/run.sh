# part 1
python DQN_problem.py --episodes 1000 --gamma 0.98 --buffer 20000 --batch 128 --decay step
# smaller gamma
python DQN_problem.py --episodes 1000 --gamma 0.5 --buffer 20000 --batch 128 --decay step
#gamma 1
python DQN_problem.py --episodes 1000 --gamma 1 --buffer 20000 --batch 128 --decay step
# smaller memory
python DQN_problem.py --episodes 1000 --gamma 0.98 --buffer 5000 --batch 128 --decay step
#bigger memory
python DQN_problem.py --episodes 1000 --gamma 0.98 --buffer 30000 --batch 128 --decay step
#Fewer Episodes
python DQN_problem.py --episodes 500 --gamma 0.98 --buffer 20000 --batch 128 --decay step

