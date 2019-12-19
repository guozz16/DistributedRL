import pypower.api
from powersys import PowerSys,Agent
import numpy as np
from numpy import flatnonzero as find
import matplotlib.pyplot as plt
AGENT = [27,23]
TRAIN = 0
HISTORY = 1
TEST = 1
RENDER = 0
TRAIN_CHANGE = 1
TEST_CHANGE = 0
TRAIN_STEP = 4000
TEST_STEP = 200
OUT_STEP = 10
RENDER_STEP = 10
CHANGE_STEP = 30
TEST_EPISODE = 100
TEST_SAVE = 1
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
	loss_reduction=[]
	info_reduction=[]
	agents = []
	env.reset()
	for i in AGENT:
		agents.append(Agent(env,i,1))
	for j in range(TEST_EPISODE):
		loss=[]
		info_loss = []
		dis = ['Traditional：','Info_theory：']
		for info_flag in range(2):
			env.reset()
			env.change(info_flag)
			initial_loss = env.loss
			if info_flag == 0:
				loss.append(initial_loss)
			else:
				info_loss.append(initial_loss)
			print(dis[info_flag])
			print('Initial Loss:' + str(initial_loss))
			for i in range(TEST_STEP):
				env.step(info_flag)
				if info_flag == 0:
					loss.append(env.getLoss())
				else:
					info_loss.append(env.getLoss())
				if i%CHANGE_STEP == 0 and TEST_CHANGE == 1:
					env.change()
				if i%RENDER_STEP == 0 and RENDER == 1:
					env.render()
			if info_flag == 0:
				_reduc = (loss[0]-loss[-1])/loss[0]
				print('Loss reduction rate:{:.2%}'.format(_reduc))
				loss_reduction.append(_reduc)
			else:
				_reduc = (info_loss[0] - info_loss[-1]) / info_loss[0]
				print('Loss reduction rate:{:.2%}'.format(_reduc))
				info_reduction.append(_reduc)
		plt.clf()
		plt.plot(loss, label= 'traditional')
		plt.plot(info_loss, label='info_theory')
		plt.legend()
		plt.savefig('F:/result/Figure'+str(j)+'.png',dpi=100)
	plt.clf()
	plt.plot(loss_reduction,label = 'traditional')
	plt.plot(info_reduction,label = 'info_theory')
	plt.legend()
	plt.savefig('result.png',dpi=100)
	for agent in agents:
		if TEST_SAVE == 1:
			agent.save()
		agent.close()