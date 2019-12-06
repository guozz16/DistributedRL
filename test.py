import pypower.api
from powersys import PowerSys,Agent
import numpy as np 
from numpy import flatnonzero as find
import matplotlib.pyplot as plt
AGENT = [23,27]
TRAIN = 0
HISTORY = 0
TEST = 1
RENDER = 0
TEST_CHANGE = 0
TRAIN_STEP = 4000
TEST_STEP = 300
OUT_STEP = 25
RENDER_STEP = 25
CHANGE_STEP = 30
# np.random.seed(1)
env = PowerSys(pypower.api.case30())
if TRAIN == 1:
	agents = []
	env.reset()
	# show topology
	env.render(1)
	#add Agent 
	for i in AGENT:
		agents.append(Agent(env,i,HISTORY))
	#train
	for i in range(TRAIN_STEP):
		if not env.step():
			if i%CHANGE_STEP == 0:
				env.change()
		else:
			env.reset()
			print('done')
		if i%OUT_STEP == 0:
			print(str(i)+':'+str(env.loss))
		if i%RENDER_STEP == 0 and RENDER == 1:
			env.render()
	for agent in agents:
		agent.save()
		agent.close()
if TEST == 1:
	agents = []
	loss=[]
	env.reset()
	env.change()
	initial_loss = env.getLoss()
	loss.append(initial_loss)
	print('Initial Loss:'+str(initial_loss))
	for i in AGENT:
		agents.append(Agent(env,i,1))
	for i in range(TEST_STEP):
		env.step()
		loss.append(env.getLoss())
		print(str(i)+':'+str(loss[-1]))
		if i%CHANGE_STEP == 0 and TEST_CHANGE == 1:
			env.change()
		if i%RENDER_STEP == 0 and RENDER == 1:
			env.render()
	for agent in agents:
		agent.close()
	print('Final loss reduction rate:{:.2%}'.format((initial_loss-loss[-1])/initial_loss))
	plt.clf()
	plt.plot(loss)
	plt.show()