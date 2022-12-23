#main in progress
python DDPG_problem.py --gamma 0.99 --buffer 30000 
#smaller gamma # in progress
python DDPG_problem.py --gamma 0.5 --buffer 30000
#big gamma
python DDPG_problem.py --gamma 1 --buffer 30000
#small memory
python DDPG_problem.py --gamma 0.99 --buffer 5000
#bigger memory
python DDPG_problem.py --gamma 0.99 --buffer 50000
