import pypower.api
from powersys import PowerSys,Agent
import numpy as np
import pandas as pd 
from numpy import flatnonzero as find
import matplotlib.pyplot as plt


class Test():
	def __init__(self,agent = [23,27]):
		self.agent = agent
	def train(self,history = 0, train_step = 2000,\
		change_step = 300, train_change = 1,\
		out_step = 20, info_flag = 0,\
		render_step = 25, render = 0):
		env = PowerSys()
		agents = []
		loss = []
		env.reset()
		# show topology
		env.render()
		#add Agent
		for i in self.agent:
			agents.append(Agent(env,i,history))
		#train
		for i in range(train_step):
			if not env.step(info_flag):
				if i%change_step == 0:
					if train_change == 1:
						env.change()
					else:
						env.reset()
						print('reset')
			else:
				env.reset()
				print('done')
				continue
			if i%out_step == 0:
				print('\n'+str(i)+':'+str(env.loss))
			if i%render_step == 0 and render == 1:
				env.render()
			loss.append(env.loss)
		for agent in agents:
			agent.save(info_flag)
			agent.close()
		return loss
	def test(self, test_episode = 10, test_step = 300, test_save = 0):
		env = PowerSys()
		loss_reduction=[]
		info_reduction=[]
		for j in range(test_episode):
			print('Episode:'+str(j))
			loss=[]
			info_loss = []
			dis = ['Traditional：','Info_theory：']
			for info_flag in range(2):
				agents = []
				env.reset()
				env.change(info_flag)
				env.render(name = "initial.png")
				initial_loss = env.loss
				if info_flag == 0:
					loss.append(initial_loss)
				else:
					info_loss.append(initial_loss)
				print(dis[info_flag])
				print('Initial Loss:' + str(initial_loss))
				for i in self.agent:
					agents.append(Agent(env,i,1,info_flag,0.95))
				for i in range(test_step):
					env.step(info_flag)
					if info_flag == 0:
						loss.append(env.getLoss())
					else:
						info_loss.append(env.getLoss())
				for agent in agents:
					if test_save == 1:
						agent.save(info_flag)
					agent.close()
				if info_flag == 0:
					_reduc = (loss[0]-loss[-1])/loss[0]
					print('\nLoss reduction rate:[{:.2%}]'.format(_reduc))
					loss_reduction.append(_reduc)
					env.render(name = "Final.png")
				else:
					_reduc = (info_loss[0] - info_loss[-1]) / info_loss[0]
					print('\nLoss reduction rate[:{:.2%}]'.format(_reduc))
					info_reduction.append(_reduc)
					env.render(name = "Info_Final.png")
			plt.clf()
			plt.plot(loss, label= 'traditional')
			plt.plot(info_loss, label='info_theory')
			plt.legend()
			plt.savefig('result/Figure'+str(j)+'.png',dpi=100)
		return loss_reduction,info_reduction

test = Test()
# test.train(history = 0,train_change = 0)
loss1 = test.train(history = 1)
loss2 = test.train(history = 1, info_flag = 1)
result = pd.DataFrame([loss1,loss2])
result.to_csv('result.csv')
print(test.test())