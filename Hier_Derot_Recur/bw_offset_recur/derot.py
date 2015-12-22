__author__ = 'QiYE'
import numpy
import h5py
import matplotlib.pyplot as plt
import cv2
from src import constants
from math import pi
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



def derot_test(dataset,bin_size,offset_rot,set_name,source_name,rot_save_name):
    path = "../../data/%s/rot/best/"%set_name
    rot_pred = numpy.load("%s%s%s.npy"%(path,dataset,rot_save_name))

    src_path = '../../data/%s/source/'%set_name
    path = '%s%s_%s%s.h5'%(src_path,dataset,set_name,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    joint_label_uvd = f['joint_label_uvd'][...]
    f.close()

    new_uvd = numpy.empty((21,2))
    rot_smooth_pred = numpy.empty((r0.shape[0],))

    num=30
    idx = numpy.random.randint(0,r0.shape[0],size=num)
    for k in xrange(num):
        i=idx[k]
        max_val = numpy.max(rot_pred[i])
        loc = numpy.where(rot_pred[i]==max_val)
        # smooth_pred[i] = (prediction[i][loc]).mean()/constants.Num_Class
        rot_smooth_pred[i] = loc[0].mean()*bin_size+offset_rot-180

        fig=plt.figure()
        ax=fig.add_subplot(231)
        ax.imshow(r0[i],'gray')
        plt.scatter(joint_label_uvd[i,0,0]*72+12,joint_label_uvd[i,0,1]*72+12)
        plt.scatter(joint_label_uvd[i,9,0]*72+12,joint_label_uvd[i,9,1]*72+12)
        ax=fig.add_subplot(232)
        ax.imshow(r1[i],'gray')
        ax=fig.add_subplot(233)
        ax.imshow(r2[i],'gray')

        M = cv2.getRotationMatrix2D((48,48),-rot_smooth_pred[i],1)
        new_r0 = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in xrange(joint_label_uvd.shape[1]):
            new_uvd[j] = numpy.dot(M,numpy.array([joint_label_uvd[i,j,0]*72+12,joint_label_uvd[i,j,1]*72+12,1]))/96

        M = cv2.getRotationMatrix2D((24,24),-rot_smooth_pred[i],1)
        new_r1 = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),-rot_smooth_pred[i],1)
        new_r2 = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)
        ax=fig.add_subplot(234)
        ax.imshow(new_r0,'gray')
        plt.scatter(new_uvd[0,0]*96,new_uvd[0,1]*96)
        plt.scatter(new_uvd[9,0]*96,new_uvd[9,1]*96)
        ax=fig.add_subplot(235)
        ax.imshow(new_r1,'gray')
        ax=fig.add_subplot(236)
        ax.imshow(new_r2,'gray')
        plt.show()
def get_rot(joint_label_uvd,setname,i,j):

    if setname=='nyu':
        vect = joint_label_uvd[:,i,0:2] - joint_label_uvd[:,j,0:2]#the index is valid for 21joints
    if setname=='msrc':
        vect = joint_label_uvd[:,j,0:2] - joint_label_uvd[:,i,0:2]#the index is valid for 21joints

    rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    loc_neg = numpy.where(vect[:,0]<0)
    rot[loc_neg] = -rot[loc_neg]
    rot = numpy.cast['float32'](rot/pi*180)
    print numpy.where(rot==180)[0].shape[0]
    rot[numpy.where(rot==180)] =179
    return rot


def rot_img(r0,r1,r2,pred_uvd, gr_uvd ,rotation):
    for i in xrange(0,gr_uvd.shape[0],1):
        M = cv2.getRotationMatrix2D((48,48),-rotation[i],1)
        r0[i] = cv2.warpAffine(r0[i],M,(96,96),borderValue=1)

        for j in xrange(gr_uvd.shape[1]):
            gr_uvd[i,j,0:2] = numpy.dot(M,numpy.array([gr_uvd[i,j,0]*72+12,gr_uvd[i,j,1]*72+12,1]))/96
            pred_uvd[i,j,0:2] = numpy.dot(M,numpy.array([pred_uvd[i,j,0]*72+12,pred_uvd[i,j,1]*72+12,1]))/96

        M = cv2.getRotationMatrix2D((24,24),-rotation[i],1)
        r1[i] = cv2.warpAffine(r1[i],M,(48,48),borderValue=1)

        M = cv2.getRotationMatrix2D((12,12),-rotation[i],1)
        r2[i] = cv2.warpAffine(r2[i],M,(24,24),borderValue=1)

    return
def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'


def get_rot_hist(x,bin_size):
    print x.shape
    # Make a normed histogram. It'll be multiplied by 100 later.
    y = plt.hist(x, bins=360/bin_size,range=(-180,180),normed=True)
    print numpy.max(y[0])*100
    print numpy.sum(y[0]*6)
    # Create the formatter using the function to_percent. This multiplies all the
    # default labels by 100, making them all percentages
    formatter = FuncFormatter(to_percent)

    # Set the formatter
    plt.xlim(xmin=-180,xmax=180)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.show()



def derot_dataset(dataset,setname,source_name,upd_pred_name):
    direct = '../../data/%s/whole/best/'%setname
    prev_jnt_path ='%s%s%s.npy'%(direct,dataset,upd_pred_name)
    pred_uvd = numpy.load(prev_jnt_path)
    pred_uvd.shape = (pred_uvd.shape[0],21,3)

    src_path = '../../data/%s/source/'%setname
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,source_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    gr_uvd = f['joint_label_uvd'][...]
    src_path = '../../data/%s/recursive/'%setname
    f_derot = h5py.File('%s%s_%s_updrot%s.h5'%(src_path,dataset,setname,source_name),'w')
    print r0.shape
    for key in f.keys():
        f.copy(key,f_derot)
    f.close()

    upd_rot = get_rot(pred_uvd,setname,0,9)

    gr_rot = get_rot(gr_uvd,setname,0,9)
    print 'rot hist bf derot'
    get_rot_hist(gr_rot,6)

    rot_img(r0,r1,r2,pred_uvd,gr_uvd,upd_rot)

    gr_rot = get_rot(gr_uvd,setname,0,9)
    print 'rot hist af derot'
    get_rot_hist(gr_rot,6)



    f_derot.create_dataset('upd_rot', data=upd_rot)
    f_derot.create_dataset('gr_uvd_derot', data=gr_uvd)
    f_derot.create_dataset('pred_uvd_derot', data=pred_uvd)
    f_derot['r0'][...]=r0
    f_derot['r1'][...]=r1
    f_derot['r2'][...]=r2
    f_derot.close()



def read_derot_dataset(dataset,setname,file_name):

    src_path = '../../data/%s/recursive/'%setname
    path = '%s%s_%s%s.h5'%(src_path,dataset,setname,file_name)
    f = h5py.File(path,'r')
    # print f.keys()
    r0 = f['r0'][...]
    r1 = f['r1'][...]
    r2= f['r2'][...]
    gr_uvd_derot = f['gr_uvd_derot'][...]*96
    pred_uvd_derot = f['pred_uvd_derot'][...]*96
    f.close()
    # vect = gr_uvd_derot[:,9,0:2] - gr_uvd_derot[:,0,0:2]#the index is valid for 21joints
    # rot =numpy.arccos(numpy.dot(vect,(0,1))/numpy.linalg.norm(vect,axis=1))
    # loc_neg = numpy.where(vect[:,0]<0)
    # rot[loc_neg] = -rot[loc_neg]
    # rot = numpy.cast['float32'](rot/2/3.1415926+0.5)
    # print 'norm to [0,1]'
    # bin = 6
    # rot = numpy.asarray(numpy.floor(rot*360),dtype='int32')
    # rot[numpy.where(rot==360)] =359
    # print numpy.min(rot)
    # print numpy.max(rot)
    # rot_bin = (rot-16)/bin
    # hist = numpy.zeros((constants.Num_Class,),dtype='int32')
    # for i in xrange(rot_bin.shape[0]):
    #     hist[rot_bin[i]] +=1
    # print hist
    num=20
    idx = numpy.random.randint(0,r0.shape[0],size=num)
    for k in xrange(num):
        i = idx[k]
        fig=plt.figure()
        # ax=fig.add_subplot(131)
        # ax.imshow(r0[i],'gray')
        plt.imshow(r0[i],'gray')
        plt.scatter(gr_uvd_derot[i,:,0],gr_uvd_derot[i,:,1])
        plt.scatter(pred_uvd_derot[i,:,0],pred_uvd_derot[i,:,1],c='r')

        # ax=fig.add_subplot(132)
        # ax.imshow(r1[i],'gray')
        # ax=fig.add_subplot(133)
        # ax.imshow(r2[i],'gray')
        plt.show()

if __name__ == "__main__":

    read_derot_dataset(dataset='test',setname='nyu',file_name='_updrot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300')
    derot_dataset(dataset='train',setname='nyu',
                  source_name='_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
                  upd_pred_name='_whole_21jnts_r012_conti_c0032_c0164_c1032_c1164_c2032_c2164_h18_h232_gm0_lm300_yt0_ep905')

