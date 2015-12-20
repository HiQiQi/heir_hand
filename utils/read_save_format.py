__author__ = 'QiYE'
import numpy
from scipy.io import  savemat,loadmat
def load(path,format):
    if format =='npy':
        return numpy.load(path)
    else:
        data = loadmat(path)
        return data['jnt']
def save(path,data,format):
    if format =='npy':
        numpy.save(path,data)
    else:
        savemat(path,{'jnt':data})

    return

