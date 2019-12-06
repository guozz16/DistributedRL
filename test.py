import pypower.api
from powersys import PowerSys,Agent
import numpy as np 

np.random.seed(1)
env = PowerSys(pypower.api.case30())
env.reset()
# env.render()
# a1 = Agent(env,23)
a1 = Agent(env,23,'Agent23.csv')
# for i in range(4000):
# 	if not env.step():
# 		print(env.loss)
# 	else:
# 		print('done')
# a1.save()
# env.reset()
env.ppc['bus'][14,3] = 5
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
env.step()
print(env.getLoss())
# a1.save()