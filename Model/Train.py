__author__ = 'QiYE'

import theano.tensor as T
import theano
import numpy
def update_params(params,grads,gamma = 0.0,
    lamda = 0.01,
    yita = 0.000):

    delta = []
    for param_i in params:
        delta.append(theano.shared(param_i.get_value(), borrow=False))
    updates = []
    for param_i, delta_i, grad_i in zip(params, delta, grads):
        updates.append((delta_i, gamma*delta_i - lamda*(yita*param_i + grad_i)))
        updates.append((param_i, param_i + gamma*delta_i - lamda*(yita*param_i+grad_i)))
    return updates

def update_params2(model,cost,learning_rate,momentum,):

    updates = []

    for param in  model.params:
      param_update = theano.shared(param.get_value()*numpy.cast[theano.config.floatX](0.))
      updates.append((param, param - learning_rate*param_update))
      updates.append((param_update, momentum*param_update + (numpy.cast[theano.config.floatX](1.) - momentum)*T.grad(cost, param)))

    return updates

def set_params(path,params):
    model_info = numpy.load(path)
    params_val = model_info[0]
    # cost_v = model_info[-1][-1]
    # print 'cost_v',cost_v
    for param_i, params_v in zip(params, params_val):
        param_i.set_value(params_v)
    return

def get_gradients(model,cost):

    return T.grad(cost, model.params)
