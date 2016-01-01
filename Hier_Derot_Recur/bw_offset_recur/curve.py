__author__ = 'QiYE'

import numpy
import cPickle
import matplotlib.pyplot as plt
from src.utils import constants
save_path = '%sdata/msrc/hier_derot_recur/bw_offset/'%constants.Data_Path
model_save_path = "%sparam_cost_uvd_bw5_r012_egoff2_c0064_h11_h22_gm0_lm3000_yt0_ep1440.npy"%save_path

model_info = numpy.load(model_save_path)
train_cost = numpy.array(model_info[-2][1:-1])
test_cost = numpy.array(model_info[-1][1:-1])


print 'train cost...', numpy.min(train_cost),numpy.max(train_cost)
print 'test cost...' ,numpy.min(test_cost),numpy.max(test_cost)

# for i in xrange(train_cost.shape[0]-1):
#     print 'epoch%d'%(i+1)
#     print 'train: %f, test: %f'%(train_cost[i],test_cost[i]), 'train decrese: %f, test decrese: %f'%(train_cost[i+1] - train_cost[i], test_cost[i+1] - test_cost[i])

max_cost = numpy.max([numpy.max(train_cost),numpy.max(test_cost)])
# min_cost = numpy.min([numpy.min(train_cost),numpy.min(test_cost)])
min_cost = max_cost*0.1
train_cost -= min_cost
train_cost = train_cost/(max_cost-min_cost)
test_cost -= min_cost
test_cost = test_cost/(max_cost-min_cost)
print numpy.max(train_cost),numpy.min(train_cost)
print numpy.max(test_cost),numpy.min(test_cost)

x_axis = train_cost.shape[0]
fig = plt.figure()
plt.ylim(ymin=0.1,ymax=1)
plt.xlim(xmin=1,xmax=x_axis)
plt.plot(numpy.arange(1,x_axis,1),train_cost[0:x_axis-1,], 'blue')
plt.plot(numpy.arange(1,x_axis,1),test_cost[0:x_axis-1,], 'red')
plt.xscale('log')
plt.yscale('log')
plt.grid('on','minor')
plt.tick_params(which='minor' )

plt.show()