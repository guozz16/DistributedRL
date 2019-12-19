import pypower.api
from powersys import PowerSys,Agent
import numpy as np
from numpy import flatnonzero as find
import matplotlib.pyplot as plt
AGENT = [27,23]
TRAIN = 0
HISTORY = 0
TEST = 1
RENDER = 0
TRAIN_CHANGE = 1
TEST_CHANGE = 0
TRAIN_STEP = 40000
TEST_STEP = 200
OUT_STEP = 25
RENDER_STEP = 25
CHANGE_STEP = 500
# np.random.seed(802)
env = PowerSys()
if TRAIN == 1:
	agents = []
	env.reset()
	# show topology
	env.render()
	#add Agent
	for i in AGENT:
		agents.append(Agent(env,i,HISTORY))
	#train
	for i in range(TRAIN_STEP):
		if not env.step():
			if i%CHANGE_STEP == 0:
				if TRAIN_CHANGE == 1:
					env.change()
				else:
					env.reset()
					print('reset')
		else:
			env.reset()
			print('done')
			continue
		if i%OUT_STEP == 0:
			print('\n'+str(i)+':'+str(env.loss))
		if i%RENDER_STEP == 0 and RENDER == 1:
			env.render()
	for agent in agents:
		agent.save()
		agent.close()
if TEST == 1:
	loss=[]
	info_loss = []
	dis = ['Traditional：','Info_theory：']
	for info_flag in range(2):
		agents = []
		env.reset()
		env.change(info_flag)
		initial_loss = env.loss
		if info_flag == 0:
			loss.append(initial_loss)
		else:
			info_loss.append(initial_loss)
		print(dis[info_flag])
		print('Initial Loss:' + str(initial_loss))
		for i in AGENT:
			agents.append(Agent(env,i,1))
		for i in range(TEST_STEP):
			env.step(info_flag)
			if info_flag == 0:
				loss.append(env.getLoss())
				print(str(i) + ':' + str(loss[-1]))
			else:
				info_loss.append(env.getLoss())
				print(str(i)+':'+str(info_loss[-1]))
			if i%CHANGE_STEP == 0 and TEST_CHANGE == 1:
				env.change()
			if i%RENDER_STEP == 0 and RENDER == 1:
				env.render()
		for agent in agents:
			agent.close()
		if info_flag == 0:
			print('Final loss reduction rate:{:.2%}'.format((loss[0]-loss[-1])/loss[0]))
		else:
			print('Final loss reduction rate:{:.2%}'.format((info_loss[0] - info_loss[-1]) / info_loss[0]))
	plt.clf()
	plt.plot(loss, label= 'traditional')
	plt.plot(info_loss, label='info_theory')
	plt.legend()
	plt.savefig("Figure_0.png",dpi=150)
	plt.show()