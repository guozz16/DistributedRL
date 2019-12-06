from RL_brain import QLearningTable 
from pypower.api import ppoption, runpf
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
	def __init__(self,PowerSys,Node,k=0):
		assert any(PowerSys.case['bus'][:,BUS_I]==Node),'There is no bus %d at powersys. '%Node
		assert any(PowerSys.case['gen'][:,GEN_BUS]==Node),'There is no gen at bus %d. '%Node
		self.powersys = PowerSys
		self.powersys.agents.append(self)
		self.bus_i = Node
		self.node = find(self.powersys.case['bus'][:,BUS_I]==Node)[0]
		self.node_g = self.powersys.b2g[self.node]
		self.type = int(self.powersys.case['bus'][self.node,BUS_TYPE])
		if k==0: # Initial Q-table
			self.brain = QLearningTable(['-1','-0.3','0','0.3','1'])
		else: # Load trained Q-table
			self.brain = QLearningTable(['-1','-0.3','0','0.3','1'],q_table='Agent'+str(self.bus_i)+'.csv') # two actions -1 for up 1 for down 0 for remain
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
		if any(self.powersys.case['branch'][:,F_BUS]==self.bus_i):
			_list = find(self.powersys.case['branch'][:,F_BUS]==self.bus_i)
			for i in _list:
				temp.append(find(self.powersys.case['bus'][:,BUS_I]==self.powersys.case['branch'][i,T_BUS])[0])
		if any(self.powersys.case['branch'][:,T_BUS]==self.bus_i):
			_list = find(self.powersys.case['branch'][:,T_BUS]==self.bus_i)
			for i in _list:
				temp.append(find(self.powersys.case['bus'][:,BUS_I]==self.powersys.case['branch'][i,F_BUS])[0])
		return temp
	# learn from last move
	def learn(self,reward):
		# learn from global reward info
		_state = self._getState()
		self.brain.learn(self.state,str(self.action),reward,_state)
		self.state = _state
		# self.save()
	# save training result
	def save(self):
		self.brain.q_table.to_csv('Agent'+str(self.bus_i)+'.csv')
	def close(self):
		self.powersys.agents.remove(self)

class PowerSys():
	def __init__(self,case):
		self.ppopt = ppoption(VERBOSE=0,OUT_ALL=0,OUT_SYS_SUM=False,\
			OUT_BUS=False,OUT_BRANCH=False) # simplify runpf out info
		self.case = case # assign case
		self.agents = []
		self.b2g={} # map bus num to gen num
		self.g2b={} # map gen num to bus num
		for g in list(range(len(self.case['gen']))):
			self.g2b[g] = find(self.case['bus'][:,BUS_I]==self.case['gen'][g,GEN_BUS])[0]
			self.b2g[self.g2b[g]] = g
		self.pv=find(self.case['bus'][:,BUS_TYPE]==PV)
		self.ref=find(self.case['bus'][:,BUS_TYPE]==REF)
		self.pq=find(self.case['bus'][:,BUS_TYPE]==PQ)

		#initialize graph layout and colormap
		g_ = nx.Graph()
		for bus in self.case['bus']:
		    g_.add_node(int(bus[BUS_I]))
		for brch in self.case['branch']:
		    g_.add_edge(int(brch[F_BUS]),int(brch[T_BUS]))
		self.pos = nx.kamada_kawai_layout(g_)
		red_ = '#ff0000'   #red
		green_ = '#00ff00'     #green
		blue_ = '#0000ff'     #blue
		self.cmap = col.LinearSegmentedColormap.from_list('cmap',[blue_,green_,red_])

	def reset(self):
		#initial pf
		self.ppc = self.case
		self.results, self.success = runpf(self.ppc,self.ppopt)
		self.loss = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
	def getLoss(self):
		self.results, self.success = runpf(self.ppc,self.ppopt)
		self.loss = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
		return self.loss
	def step(self,k=0):
		d_=False
		for agent in self.agents:
			agent.choose()
		for agent in self.agents:
			agent.move()
			if k==1:
				print('Agent'+str(agent.bus_i)+':'+str(agent.action))
		#update power flow
		self.results, self.success = runpf(self.ppc,self.ppopt)
		loss_ = sum(self.results['gen'][:,PG])-sum(self.results['bus'][:,PD])
		if loss_ <= self.loss:
			r_ = 1
		else:
			r_ = 0
		self.loss = loss_
		# check for voltage violation
		bi_max = find(self.results['bus'][:,VM]-0.001>self.results['bus'][:,VMAX])
		bi_min = find(self.results['bus'][:,VM]+0.001<self.results['bus'][:,VMIN])
		if bi_max.size is not 0:
			r_ = 0
			print('Voltage violated bus list',bi_max)
			d_ = True
		elif bi_min.size is not 0:
			r_ = 0
			print('Voltage violated bus list',bi_min)
			d_ = True
		#check for P violation
		if any(self.results['gen'][:,PG]>self.results['gen'][:,PMAX]) or any(self.results['gen'][:,PG]<self.results['gen'][:,PMIN]):
			r_ = 0
			d_ = True
		#check for Q violation
		if any(self.results['gen'][:,QG]>self.results['gen'][:,QMAX]) or any(self.results['gen'][:,QG]<self.results['gen'][:,QMIN]):
			r_ = 0
			d_ = True
		for agent in self.agents:
			agent.learn(r_)
		return d_
	def change(self):
		#random load change
		temp_bus = np.random.choice(self.ppc['bus'][:,0])
		temp_i = find(self.ppc['bus'][:,0]==temp_bus)[0]
		temp_change = np.random.randn()
		temp_load = self.ppc['bus'][temp_i,3]+temp_change
		print('Bus %d change reactive load from %.2f to %.2f. '%(int(temp_bus),self.ppc['bus'][temp_i,3],temp_load))
		self.ppc['bus'][temp_i,3] = temp_load
	def render(self,k=0):
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
		plt.savefig("Figure_1.png",dpi=150)
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
		plt.savefig("Figure_2.png",dpi=150)
		if k==1:
			plt.show()

