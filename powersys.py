from RL_brain import QLearningTable 
from pypower.api import ppoption, runpf
from pypower.api import case30 as case
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, PV, REF, BUS_I, VMAX, VMIN
from pypower.idx_brch import PF, QF, F_BUS, T_BUS
from pypower.idx_gen import PG, QG, VG, PMAX, PMIN, QMAX, QMIN, GEN_BUS, GEN_STATUS
import networkx as nx 
import numpy as np 
from numpy import r_
from numpy import flatnonzero as find
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import matplotlib.colors as col 
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

VUNIT = 0.001 #p.u.
QUNIT = 0.01 #p.u.
class Agent():
	def __init__(self,PowerSys,Node,k=0,info_flag=0,e_greedy = 0.9):
		assert any(PowerSys.ppc['bus'][:,BUS_I]==Node),'There is no bus %d at powersys. '%Node
		assert any(PowerSys.ppc['gen'][:,GEN_BUS]==Node),'There is no gen at bus %d. '%Node
		self.powersys = PowerSys
		self.powersys.agents.append(self)
		self.bus_i = Node
		self.node = find(self.powersys.ppc['bus'][:,BUS_I]==Node)[0]
		self.node_g = self.powersys.b2g[self.node]
		self.type = int(self.powersys.ppc['bus'][self.node,BUS_TYPE])
		if k==0: # Initial Q-table
			self.brain = QLearningTable(['-1','-0.3','0','0.3','1'],e_greedy = e_greedy)
		elif info_flag == 0: # Load trained Q-table
			self.brain = QLearningTable(['-1','-0.3','0','0.3','1'],\
			q_table='Agent'+str(self.bus_i)+'.csv',e_greedy = e_greedy) # -1 for up 1 for down 0 for remain
		else:
			self.brain = QLearningTable(['-1','-0.3','0','0.3','1'],\
			q_table='Info_Agent'+str(self.bus_i)+'.csv',e_greedy = e_greedy)
		self.nodes = self._getNeighbour()
		self.state = self._getState()

	# Get local state including VM VA of neighbour nodes
	def _getState(self):
		temp = '%.3f'%self.powersys.results['bus'][self.node,VM]+'|'+\
		'%.1f'%self.powersys.results['bus'][self.node,VA]+'|'
		for i in self.nodes:
			temp = temp + \
			'%.3f'%self.powersys.results['bus'][i,VM]+'|'\
			'%.1f'%self.powersys.results['bus'][i,VA]+'|'
		return temp
	# Choose action based on local state
	def choose(self):
		self.action = self.brain.choose_action(self.state)
	# Execute chosen action
	def move(self):
		if self.type == 1:
			self.powersys.ppc['gen'][self.node_g,QG] += float(self.action)*QUNIT
		else :
			self.powersys.ppc['gen'][self.node_g,VG] += float(self.action)*VUNIT
	# generate neighbour node set
	def _getNeighbour(self):
		# Get neighbour nodes based on topology
		temp = []
		_list = []
		if any(self.powersys.ppc['branch'][:,F_BUS]==self.bus_i):
			_list = find(self.powersys.ppc['branch'][:,F_BUS]==self.bus_i)
			for i in _list:
				temp.append(find(self.powersys.ppc['bus'][:,BUS_I]==self.powersys.ppc['branch'][i,T_BUS])[0])
		if any(self.powersys.ppc['branch'][:,T_BUS]==self.bus_i):
			_list = find(self.powersys.ppc['branch'][:,T_BUS]==self.bus_i)
			for i in _list:
				temp.append(find(self.powersys.ppc['bus'][:,BUS_I]==self.powersys.ppc['branch'][i,F_BUS])[0])
		return temp
	# learn from last move
	def learn(self,reward):
		# learn from global reward info
		_state = self._getState()
		self.brain.learn(self.state,str(self.action),reward,_state)
		self.state = _state
	# learn from last move
	def info_learn(self,reward):
		# learn from global reward info
		_state = self._getState()
		self.brain.info_learn(self.state,str(self.action),reward,_state)
		self.state = _state	
	# save training result
	def save(self,info_flag=0):
		if info_flag == 0:
			self.brain.q_table.to_csv('Agent'+str(self.bus_i)+'.csv')
		else:
			self.brain.q_table.to_csv('Info_Agent'+str(self.bus_i)+'.csv')
	def close(self):
		self.powersys.agents.remove(self)

class PowerSys():
	def __init__(self):
		self.ppopt = ppoption(VERBOSE=0,OUT_ALL=0,OUT_SYS_SUM=False,\
			OUT_BUS=False,OUT_BRANCH=False) # simplify runpf out info
		self.ppc = case()
		self.agents = []
		self.b2g={} # map bus num to gen num
		self.g2b={} # map gen num to bus num
		for g in list(range(len(self.ppc['gen']))):
			self.g2b[g] = find(self.ppc['bus'][:,BUS_I]==self.ppc['gen'][g,GEN_BUS])[0]
			self.b2g[self.g2b[g]] = g
		self.pv=find(self.ppc['bus'][:,BUS_TYPE]==PV)
		self.ref=find(self.ppc['bus'][:,BUS_TYPE]==REF)
		self.pq=find(self.ppc['bus'][:,BUS_TYPE]==PQ)

		#initialize graph layout and colormap
		g_ = nx.Graph()
		for bus in self.ppc['bus']:
		    g_.add_node(int(bus[BUS_I]))
		for brch in self.ppc['branch']:
		    g_.add_edge(int(brch[F_BUS]),int(brch[T_BUS]))
		self.pos = nx.kamada_kawai_layout(g_)
		red_ = '#ff0000'   #red
		green_ = '#00ff00'     #green
		blue_ = '#0000ff'     #blue
		self.cmap = col.LinearSegmentedColormap.from_list('cmap',[blue_,green_,red_])

	def reset(self):
		#initial pf
		self.ppc = case()
		self.results, self.success = runpf(self.ppc,self.ppopt)
		self.loss = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
	def getLoss(self):
		self.results, self.success = runpf(self.ppc,self.ppopt)
		self.loss = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
		return self.loss
	def step(self,info_flag=0):
		d_=False
		for agent in self.agents:
			agent.choose()
		for agent in self.agents:
			agent.move()
			# print('Agent'+str(agent.bus_i)+':'+str(agent.action))
		#update power flow
		self.results, self.success = runpf(self.ppc,self.ppopt)
		loss_ = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
		# r_ = np.log2(self.loss/loss_)
		r_ = self.loss-loss_
		# r_ = round(self.loss,4) - round(loss_,4)
		# if loss_ < self.loss:
		# 	r_ = 1
		# elif loss_>self.loss:
		# 	r_ = -1
		# else:
		# 	r_ = 0
		self.loss = loss_
		# check for voltage violation
		bi_max = find(self.results['bus'][:,VM]-0.001>self.results['bus'][:,VMAX])
		bi_min = find(self.results['bus'][:,VM]+0.001<self.results['bus'][:,VMIN])
		if bi_max.size is not 0:
			r_ = -1
			print('Voltage violated bus list',bi_max)
			d_ = True
		elif bi_min.size is not 0:
			r_ = -1
			print('Voltage violated bus list',bi_min)
			d_ = True
		#check for P violation
		if any(self.results['gen'][:,PG]>self.results['gen'][:,PMAX]) or any(self.results['gen'][:,PG]<self.results['gen'][:,PMIN]):
			r_ = -1
			d_ = True
		#check for Q violation
		if any(self.results['gen'][:,QG]>self.results['gen'][:,QMAX]) or any(self.results['gen'][:,QG]<self.results['gen'][:,QMIN]):
			r_ = -1
			d_ = True
		for agent in self.agents:
			if info_flag is 0:
				agent.learn(r_)
			else:
				agent.info_learn(r_)
		return d_
	def change(self,k=0):
		#random load change
		if k == 0:
			temp_bus = np.random.choice(self.ppc['bus'][:,0])
			self.temp_i = find(self.ppc['bus'][:,0]==temp_bus)[0]
			temp_change = np.random.randn()
			self.temp_load = round(self.ppc['bus'][self.temp_i,QD]+temp_change,2)
			print('Change reactive load from %.2f to %.2f [Bus %d] '%(self.ppc['bus'][self.temp_i,QD],self.temp_load,int(temp_bus)))
			self.ppc['bus'][self.temp_i,QD] = self.temp_load
		else:
			self.ppc['bus'][self.temp_i,QD] = self.temp_load
		self.results, self.success = runpf(self.ppc,self.ppopt)
		self.loss = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
	def render(self,k=0,name1="Figure_1.png",name2="Figure_2.png"):
		#visualize power flow
		g1 = nx.DiGraph() # P flow
		g2 = nx.DiGraph() # Q flow
		P_colors = []
		Q_colors = []
		V_colors = []
		theta_colors = []
		for brch in self.results['branch']:
		    P_colors.append(abs(brch[PF]))
		    if brch[PF]>0:
		        g1.add_edge(int(brch[F_BUS]),int(brch[T_BUS]))
		    else:
		        g1.add_edge(int(brch[T_BUS]),int(brch[F_BUS]))
		for brch in self.results['branch']:
		    Q_colors.append(abs(brch[QF]))
		    if brch[QF]>0:
		        g2.add_edge(int(brch[F_BUS]),int(brch[T_BUS]))
		    else:
		        g2.add_edge(int(brch[T_BUS]),int(brch[F_BUS]))
		for bus in self.results['bus']:
		    V_colors.append(bus[VM])
		    theta_colors.append(bus[VA])
		edges=nx.draw_networkx_edges(g1, self.pos, width=2, alpha=0.5,edge_color=P_colors,
		    edge_cmap=self.cmap,arrows=False)
		plt.clf()
		plt.axis('off')
		plt.title("Vmax:"+str(round(max(self.results['bus'][:,VM]),2))+"   "+
		    "Vmin:"+str(round(min(self.results['bus'][:,VM]),2))+"   "+
            "Loss:"+str(round(sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD]),3)))
		cbar=plt.colorbar(edges)
		cbar.set_label('$P(MW)$')
		nx.draw_networkx_nodes(g1,self.pos,nodelist=list(self.ppc['bus'][:,BUS_I].astype(int)),
		    cmap=self.cmap,node_color=theta_colors,node_size=300,alpha=0.8)
		nx.draw_networkx_edges(g1, self.pos, width=2, alpha=0.5,edge_color=P_colors,
		    edge_cmap=self.cmap,arrows=True)
		nx.draw_networkx_labels(g1, self.pos)
		plt.savefig(name1,dpi=150)
		# plt.show()
		edges=nx.draw_networkx_edges(g2, self.pos, width=2, alpha=0.5,edge_color=Q_colors,
		    edge_cmap=self.cmap,arrows=False)
		plt.clf()
		plt.axis('off')
		plt.title("Vmax:"+str(round(max(self.results['bus'][:,VM]),2))+"   "+
		    "Vmin:"+str(round(min(self.results['bus'][:,VM]),2))+"   "+
            "Loss:"+str(round(sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD]),3)))
		cbar=plt.colorbar(edges)
		cbar.set_label('$Q(Mvar)$')
		nx.draw_networkx_nodes(g2,self.pos,nodelist=list(self.ppc['bus'][:,BUS_I].astype(int)),
		    cmap=self.cmap,node_color=V_colors,node_size=300,alpha=0.8)
		nx.draw_networkx_edges(g2, self.pos, width=2, alpha=0.5,edge_color=Q_colors,
		    edge_cmap=self.cmap,arrows=True)
		nx.draw_networkx_labels(g2, self.pos)
		plt.savefig(name2,dpi=150)
		if k==1:
			plt.show()

