__author__ = 'QiYE'
import numpy
import matplotlib.pyplot as plt
import sys
from show_statistics import show_hist
def offset_to_abs(off_uvd, pre_uvd,patch_size=44,offset_depth_range=1.0,hand_width=96):

    if len(off_uvd.shape)<3:
        off_uvd[:,0:2] = (off_uvd[:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,0:2] = pre_uvd[:,0:2]+off_uvd[:,0:2]
        off_uvd[:,2] = (off_uvd[:,2]-0.5)*offset_depth_range
        predict_uvd[:,2] = pre_uvd[:,2]+off_uvd[:,2]
        return predict_uvd
    else:
        pre_uvd.shape=(pre_uvd.shape[0],1,pre_uvd.shape[-1])
        off_uvd[:,:,0:2] = (off_uvd[:,:,0:2]*patch_size -patch_size/2 )/hand_width
        # off_uvd[:,0:2] = (off_uvd[:,0:2]*72+12)/24
        predict_uvd= numpy.empty_like(off_uvd)
        predict_uvd[:,:,0:2] = pre_uvd[:,:,0:2]+off_uvd[:,:,0:2]
        off_uvd[:,:,2] = (off_uvd[:,:,2]-0.5)*offset_depth_range
        predict_uvd[:,:,2] = pre_uvd[:,:,2]+off_uvd[:,:,2]
        return predict_uvd
def norm_offset_uvd(cur_uvd,prev_uvd,offset_depth_range=1.0,hand_width=96,patch_size=36):

    if len(cur_uvd.shape)>2:
        off_uvd = cur_uvd-prev_uvd.reshape((prev_uvd.shape[0],1,prev_uvd.shape[1]))
        print 'bf norm max, min'
        print numpy.max(off_uvd[:,:,0]), numpy.min(off_uvd[:,:,0])
        print numpy.max(off_uvd[:,:,1]), numpy.min(off_uvd[:,:,1])
        print numpy.max(off_uvd[:,:,2]), numpy.min(off_uvd[:,:,2])

        # show_hist(off_uvd[:,:,0])
        # show_hist(off_uvd[:,:,1])
        # show_hist(off_uvd[:,:,2])
        print 'sample whose joint is beyonde the set offset depth range', numpy.where(numpy.abs(off_uvd[:,:,2])>offset_depth_range/2)[0].shape[0]
        off_uvd[:,:,0:2] = (off_uvd[:,:,0:2]*hand_width+patch_size/2)/patch_size
        off_uvd[:,:,2] = off_uvd[:,:,2]/offset_depth_range+0.5
        print 'af norm max, min'
        print numpy.max(off_uvd[:,:,0]), numpy.min(off_uvd[:,:,0])
        print numpy.max(off_uvd[:,:,1]), numpy.min(off_uvd[:,:,1])
        print numpy.max(off_uvd[:,:,2]), numpy.min(off_uvd[:,:,2])

    else:
        off_uvd = cur_uvd-prev_uvd
        print 'bf norm max, min'
        print numpy.max(off_uvd[:,0]), numpy.min(off_uvd[:,0])
        print numpy.max(off_uvd[:,1]), numpy.min(off_uvd[:,1])
        print numpy.max(off_uvd[:,2]), numpy.min(off_uvd[:,2])

        # show_hist(off_uvd[:,0])
        # show_hist(off_uvd[:,1])
        # show_hist(off_uvd[:,2])
        print 'sample whose joint is beyonde the set offset depth range', numpy.where(numpy.abs(off_uvd[:,2])>offset_depth_range/2)[0].shape[0]
        off_uvd[:,0:2] = (off_uvd[:,0:2]*hand_width+patch_size/2)/patch_size
        off_uvd[:,2] = off_uvd[:,2]/offset_depth_range+0.5
        print 'af norm max, min'
        print numpy.max(off_uvd[:,0]), numpy.min(off_uvd[:,0])
        print numpy.max(off_uvd[:,1]), numpy.min(off_uvd[:,1])
        print numpy.max(off_uvd[:,2]), numpy.min(off_uvd[:,2])
    return off_uvd

def patch(r0_patch,r1_patch,r2_patch,new_i,ori_i,r_center,c_center,r0,r1,r2,patch_size):
        if ((r_center-patch_size/2)>0) and ((r_center+patch_size/2) < patch_size ) and ((c_center-patch_size/2)>0) and((c_center+patch_size/2)<patch_size):
            r0_patch_tmp=r0[ori_i,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
        else:
            border=80
            r0_tmp = numpy.lib.pad(r0[ori_i], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_r_c_center = border+c_center
            r0_patch_tmp=r0_tmp[new_r_center-patch_size/2:new_r_center+patch_size/2,new_r_c_center-patch_size/2:new_r_c_center+patch_size/2]
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(center_uvd[i,0]*hand_width+pad_width,center_uvd[i,1]*hand_width+pad_width)
        #
        # plt.figure()
        # plt.imshow(r0_patch_tmp,'gray')
        # plt.show()
        mask=numpy.where(r0_patch_tmp<1)
        if mask[0].shape[0]!=0:
            dmax = numpy.max(r0_patch_tmp[mask])
            dmin = numpy.min(r0_patch_tmp[mask])
            if dmax == dmin:
                print ori_i,' dmax == dmin'
                r0_patch[new_i]=r0_patch_tmp
            else:
                r0_patch[new_i][mask] = (r0_patch_tmp[mask] -dmin)/(dmax - dmin)
        else:
            r0_patch[new_i]=r0_patch_tmp

        r_center/=2
        c_center/=2
        ratio=4
        if ((r_center-patch_size/ratio)>0) and ((r_center+patch_size/ratio) < patch_size/ratio*2 ) and ((c_center-patch_size/ratio)>0) and((c_center+patch_size/ratio)<patch_size/ratio*2):
            r1_patch_tmp=r1[ori_i,r_center-patch_size/ratio:r_center+patch_size/ratio,c_center-patch_size/ratio:c_center+patch_size/ratio]
        else:
            border=40
            r1_tmp = numpy.lib.pad(r1[ori_i], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_c_center = border+c_center
            r1_patch_tmp=r1_tmp[new_r_center-patch_size/ratio:new_r_center+patch_size/ratio,new_c_center-patch_size/ratio:new_c_center+patch_size/ratio]

        mask=numpy.where(r1_patch_tmp<1)
        if mask[0].shape[0]!=0:
            dmax = numpy.max(r1_patch_tmp[mask])
            dmin = numpy.min(r1_patch_tmp[mask])
            if dmax == dmin:
                print ori_i,' dmax == dmin r1'
                r1_patch[new_i]=r1_patch_tmp
            else:
                r1_patch[new_i][mask] = (r1_patch_tmp[mask] -dmin)/(dmax-dmin)
        else:
            r1_patch[new_i]=r1_patch_tmp


        r_center/=2
        c_center/=2
        ratio=8
        if ((r_center-patch_size/ratio)>0) and ((r_center+patch_size/ratio) < patch_size/ratio*2 ) and ((c_center-patch_size/ratio)>0) and((c_center+patch_size/ratio)<patch_size/ratio*2):
            r2_patch_tmp=r2[ori_i,r_center-patch_size/ratio:r_center+patch_size/ratio,c_center-patch_size/ratio:c_center+patch_size/ratio]
        else:
            border=20
            r2_tmp = numpy.lib.pad(r2[ori_i], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_c_center = border+c_center
            r2_patch_tmp=r2_tmp[new_r_center-patch_size/ratio:new_r_center+patch_size/ratio,new_c_center-patch_size/ratio:new_c_center+patch_size/ratio]

        mask=numpy.where(r2_patch_tmp<1)
        if mask[0].shape[0] != 0:
            dmax = numpy.max(r2_patch_tmp[mask])
            dmin = numpy.min(r2_patch_tmp[mask])
            if dmax == dmin:
                print ori_i,' dmax == dmin r2'
                r2_patch[new_i]=r2_patch_tmp
            else:
                r2_patch[new_i][mask] = (r2_patch_tmp[mask] -dmin)/(dmax-dmin)
        else:
            r2_patch[new_i]=r2_patch_tmp
        return

def conv_bw_patch(i,r0_patch,r1_patch,r2_patch,uvd,r0,r1,r2,patch_size):

    num_ft=r0_patch.shape[1]
    img_size=r0.shape[-1]

    r_center = int(uvd[0,i,1]*img_size)
    c_center = int(uvd[0,i,0]*img_size)

    for j in xrange(num_ft):
        if ((r_center-patch_size/2)>0) and ((r_center+patch_size/2) < img_size ) and ((c_center-patch_size/2)>0) and((c_center+patch_size/2)<img_size):
            r0_patch[i,j]=r0[i,j,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
        else:
            border=patch_size
            r0_tmp = numpy.lib.pad(r0[i,j], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_r_c_center = border+c_center
            r0_patch[i,j]=r0_tmp[new_r_center-patch_size/2:new_r_center+patch_size/2,new_r_c_center-patch_size/2:new_r_c_center+patch_size/2]

    # fig = plt.figure()
    # ax= fig.add_subplot(231)
    # ax.imshow(r0[i,0],'gray')
    # plt.scatter(c_center,r_center)
    # ax= fig.add_subplot(234)
    # ax.imshow(r0_patch[i,0],'gray')

    img_size=r1.shape[-1]
    r_center = int(uvd[1,i,1]*img_size)
    c_center = int(uvd[1,i,0]*img_size)
    for j in xrange(num_ft):
        if ((r_center-patch_size/2)>0) and ((r_center+patch_size/2) < img_size ) and ((c_center-patch_size/2)>0) and((c_center+patch_size/2)<img_size):
            r1_patch[i,j]=r1[i,j,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
        else:
            border=patch_size
            r1_tmp = numpy.lib.pad(r1[i,j], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_c_center = border+c_center
            r1_patch[i,j]=r1_tmp[new_r_center-patch_size/2:new_r_center+patch_size/2,new_c_center-patch_size/2:new_c_center+patch_size/2]

    # ax=fig.add_subplot(232)
    # ax.imshow(r1[i,0],'gray')
    # plt.scatter(c_center,r_center)
    # ax=fig.add_subplot(235)
    # ax.imshow(r1_patch[i,0],'gray')

    img_size=r2.shape[-1]
    r_center = int(uvd[2,i,1]*img_size)
    c_center = int(uvd[2,i,0]*img_size)
    patch_size=patch_size/2
    for j in xrange(num_ft):
        if ((r_center-patch_size/2)>0) and ((r_center+patch_size/2) < img_size ) and ((c_center-patch_size/2)>0) and((c_center+patch_size/2)<img_size):
            r2_patch[i,j]=r2[i,j,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
        else:
            border=patch_size
            r2_tmp = numpy.lib.pad(r2[i,j], ((border,border),(border,border)), 'constant',constant_values=1)
            new_r_center = border+r_center
            new_c_center = border+c_center
            r2_patch[i,j]=r2_tmp[new_r_center-patch_size/2:new_r_center+patch_size/2,new_c_center-patch_size/2:new_c_center+patch_size/2]

    # ax=fig.add_subplot(233)
    # ax.imshow(r2[i,0],'gray')
    # plt.scatter(c_center,r_center)
    # ax=fig.add_subplot(236)
    # ax.imshow(r2_patch[i,0],'gray')
    # plt.show()
    return

def crop_bw_ego_conv_patch(r0,r1,r2,uvd,patch_size):
    batch_size = r0.shape[0]
    c1=r0.shape[1]
    r0_patch = numpy.empty((6,batch_size,c1,patch_size,patch_size),dtype='float32')
    r1_patch = numpy.empty((6, batch_size,c1,patch_size,patch_size),dtype='float32')
    r2_patch = numpy.empty((6,batch_size,c1,patch_size/2,patch_size/2),dtype='float32')

    for i in xrange(r0.shape[0]):
        for j in xrange(6):
            conv_bw_patch(i=i,r0_patch=r0_patch[j,:,:,:,:],r1_patch=r1_patch[j,:,:,:,:],r2_patch=r2_patch[j,:,:,:,:],uvd=uvd[:,:,j,:],r0=r0,r1=r1,r2=r2,patch_size=patch_size)
    # for i in xrange(0,10,1):
    #     fig=plt.figure()
    #     ax=fig.add_subplot(131)
    #     ax.imshow(r0_patch[j,i,0],'gray')
    #
    #     ax=fig.add_subplot(132)
    #     ax.imshow(r1_patch[j,i,0],'gray')
    #
    #     ax=fig.add_subplot(133)
    #     ax.imshow(r1_patch[j,i,0],'gray')
    #     plt.show()
    return r0_patch,r1_patch,r2_patch




def crop_patch(center_uvd,r0,r1,r2,patch_size=24,patch_pad_width=4,hand_width=72,pad_width=12):
    """do sth"""
    r0_patch= numpy.ones((r0.shape[0],patch_size,patch_size),dtype='float32')
    r1_patch= numpy.ones((r1.shape[0],patch_size/2,patch_size/2),dtype='float32')
    r2_patch= numpy.ones((r2.shape[0],patch_size/4,patch_size/4),dtype='float32')


    for i in xrange(r0.shape[0]):
        r_center = int(center_uvd[i,1]*hand_width+pad_width)
        c_center = int(center_uvd[i,0]*hand_width+pad_width)
        patch(r0_patch,r1_patch,r2_patch,i,i,r_center,c_center,r0,r1,r2,patch_size)

    border=patch_pad_width
    r0_patch = numpy.lib.pad(r0_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r1_patch = numpy.lib.pad(r1_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r2_patch = numpy.lib.pad(r2_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    # i=1000
    # plt.figure()
    # plt.imshow(r0_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r1_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r2_patch[i],'gray')
    # plt.show()
    return r0_patch,r1_patch,r2_patch

def random_uvd(prev_uvd_pred,prev_uvd_gr,num_enlarge=1):
    # print prev_uvd_pred,prev_uvd_gr
    # print ((prev_uvd_pred+prev_uvd_gr)/2)
    if num_enlarge==1:
        return ((prev_uvd_pred+prev_uvd_gr)/2).reshape((1,3))

    if num_enlarge==2:
        replica_uvd = numpy.empty((num_enlarge,3))
        replica_uvd[0] = (prev_uvd_pred+prev_uvd_gr)/2
        replica_uvd[1]=prev_uvd_gr
        return replica_uvd

    if num_enlarge==3:
        replica_uvd = numpy.empty((num_enlarge,3))
        replica_uvd[0] = (prev_uvd_pred+prev_uvd_gr)/2
        replica_uvd[1]=prev_uvd_gr
        replica_uvd[2]=(-prev_uvd_pred+3*prev_uvd_gr)/2
        # print prev_uvd_pred
        # print replica_uvd
        return replica_uvd

def create_replica(loc,prev_uvd_pred,prev_uvd_gr,num_enlarge,r0,r1,r2,r0_patch,r1_patch,r2_patch,cur_uvd,new_prev_uvd_pred,new_cur_uvd_gr,hand_width,pad_width,patch_size):

    for i in xrange(loc.shape[0]):
        relpica_uvd = random_uvd(prev_uvd_pred[loc[i]],prev_uvd_gr[loc[i]],num_enlarge)

        for j in xrange(num_enlarge):
            r_center = int(relpica_uvd[j,1]*hand_width+pad_width)
            c_center = int(relpica_uvd[j,0]*hand_width+pad_width)
            ori_idx = loc[i]
            new_idx = i*num_enlarge+j

            new_prev_uvd_pred[new_idx]=relpica_uvd[j]
            new_cur_uvd_gr[new_idx]=cur_uvd[ori_idx]
            patch(r0_patch,r1_patch,r2_patch,new_idx,ori_idx,r_center,c_center,r0,r1,r2,patch_size)

    for i in xrange(loc.shape[0]*num_enlarge,loc.shape[0]*num_enlarge+r0.shape[0],1):
            ori_idx = i-loc.shape[0]*num_enlarge
            new_idx = i
            r_center = int(prev_uvd_pred[ori_idx,1]*hand_width+pad_width)
            c_center = int(prev_uvd_pred[ori_idx,0]*hand_width+pad_width)
            new_prev_uvd_pred[new_idx]=prev_uvd_pred[ori_idx]
            new_cur_uvd_gr[new_idx]=cur_uvd[ori_idx]

            patch(r0_patch,r1_patch,r2_patch,new_idx,ori_idx,r_center,c_center,r0,r1,r2,patch_size)
    return


def crop_patch_enlarge(cur_uvd,prev_uvd_pred,prev_uvd_gr,r0,r1,r2,num_enlarge=3,patch_size=24,patch_pad_width=4,hand_width=72,pad_width=12,batch_size=100):
    """do sth"""
    dist=numpy.sqrt(numpy.sum((prev_uvd_pred-prev_uvd_gr)**2,axis=-1))
    dist_mean= numpy.mean(dist)
    loc = numpy.where(dist>dist_mean)
    print 'err dist>dist_mean',loc[0].shape[0],'/',cur_uvd.shape[0]

    new_num_sumple = prev_uvd_pred.shape[0]+loc[0].shape[0]*num_enlarge
    if new_num_sumple%batch_size !=0:
        new_num_sumple= (batch_size-new_num_sumple%batch_size )+new_num_sumple

    new_prev_uvd_pred = numpy.empty((new_num_sumple,3),dtype='float32')
    if len(cur_uvd.shape)>2:
        new_cur_uvd_gr = numpy.empty((new_num_sumple,cur_uvd.shape[1],3),dtype='float32')
    else:
        new_cur_uvd_gr = numpy.empty((new_num_sumple,3),dtype='float32')

    r0_patch= numpy.ones((new_num_sumple,patch_size,patch_size),dtype='float32')
    r1_patch= numpy.ones((new_num_sumple,patch_size/2,patch_size/2),dtype='float32')
    r2_patch= numpy.ones((new_num_sumple,patch_size/4,patch_size/4),dtype='float32')

    create_replica(r0_patch=r0_patch,r1_patch=r1_patch,r2_patch=r2_patch,
                   new_prev_uvd_pred=new_prev_uvd_pred,new_cur_uvd_gr=new_cur_uvd_gr,
                   cur_uvd=cur_uvd,prev_uvd_pred=prev_uvd_pred,prev_uvd_gr=prev_uvd_gr,
                   loc=loc[0],num_enlarge=num_enlarge,
                   r0=r0,r1=r1,r2=r2,hand_width=hand_width,pad_width=pad_width,patch_size=patch_size)

    last_idx = (batch_size-new_num_sumple%batch_size )
    rand_idx = numpy.random.randint(low=0,high=new_num_sumple-last_idx-1,size=last_idx)
    new_prev_uvd_pred[new_num_sumple-last_idx:new_num_sumple] = new_prev_uvd_pred[rand_idx]
    new_cur_uvd_gr[new_num_sumple-last_idx:new_num_sumple] = new_cur_uvd_gr[rand_idx]

    r0_patch[new_num_sumple-last_idx:new_num_sumple]= r0_patch[rand_idx]
    r1_patch[new_num_sumple-last_idx:new_num_sumple]= r1_patch[rand_idx]
    r2_patch[new_num_sumple-last_idx:new_num_sumple]= r2_patch[rand_idx]

    border=patch_pad_width
    r0_patch = numpy.lib.pad(r0_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r1_patch = numpy.lib.pad(r1_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r2_patch = numpy.lib.pad(r2_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    # # i=0
    # plt.figure()
    # plt.imshow(r0_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r1_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r2_patch[i],'gray')
    # plt.show()
    return r0_patch,r1_patch,r2_patch,new_cur_uvd_gr,new_prev_uvd_pred
"""


def create_replica2(ratio_range,num_enlarge,prev_uvd_pred,prev_uvd_gr,r0,r1,r2,cur_uvd,hand_width,pad_width,patch_size):
    dist=numpy.sqrt(numpy.sum((prev_uvd_pred-prev_uvd_gr)**2,axis=-1))
    dist_mean= numpy.mean(dist)
    if len(ratio_range)==1:
        loc = numpy.where((dist>dist_mean*ratio_range[0]) )
    else:
        loc = numpy.where((dist>dist_mean*ratio_range[0]) & (dist < dist_mean*ratio_range[1]))
    print 'dist bw pred and gr of prev jnt',loc[0].shape[0]

    new_num_sumple = loc[0].shape[0]*num_enlarge

    new_prev_uvd_pred = numpy.empty((new_num_sumple,3),dtype='float32')
    new_cur_uvd_gr = numpy.empty((new_num_sumple,6,3),dtype='float32')

    r0_patch= numpy.ones((new_num_sumple,patch_size,patch_size),dtype='float32')
    r1_patch= numpy.ones((new_num_sumple,patch_size/2,patch_size/2),dtype='float32')
    r2_patch= numpy.ones((new_num_sumple,patch_size/4,patch_size/4),dtype='float32')
    for i in xrange(loc.shape[0]):
        relpica_uvd = random_uvd(dist[i],prev_uvd_pred[loc[i]],prev_uvd_gr[loc[i]],num_enlarge)

        for j in xrange(num_enlarge):
            r_center = int(relpica_uvd[j,1]*hand_width+pad_width)
            c_center = int(relpica_uvd[j,0]*hand_width+pad_width)
            ori_idx = loc[i]
            new_idx = i*num_enlarge+j

            new_prev_uvd_pred[new_idx]=relpica_uvd[j]
            new_cur_uvd_gr[new_idx]=cur_uvd[ori_idx]

            r0_patch_tmp=r0[ori_idx,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
            # plt.figure()
            # plt.imshow(r0[ori_idx],'gray')
            # plt.scatter(relpica_uvd[j,0]*hand_width+pad_width,relpica_uvd[j,1]*hand_width+pad_width)
            # plt.scatter(prev_uvd_pred[loc[i],0]*hand_width+pad_width,prev_uvd_pred[loc[i],1]*hand_width+pad_width,c='g')
            # plt.scatter(prev_uvd_gr[loc[i],0]*hand_width+pad_width,prev_uvd_gr[loc[i],1]*hand_width+pad_width,c='r')
            # plt.figure()
            # plt.imshow(r0_patch_tmp,'gray')
            # plt.show()
            mask=numpy.where(r0_patch_tmp<1)
            if mask[0].shape[0]==0:
                print ori_idx
                r_center = int(hand_width/2+pad_width)
                c_center = int(hand_width/2+pad_width)
                relpica_uvd[j,1]=0.5
                relpica_uvd[j,0]=0.5

                r0_patch_tmp=r0[ori_idx,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
            mask=numpy.where(r0_patch_tmp<1)
            dmax = numpy.max(r0_patch_tmp[mask])
            dmin = numpy.min(r0_patch_tmp)

            r0_patch[new_idx][mask] = (r0_patch_tmp[mask] -dmin)/(dmax - dmin)


            r_center/=2
            c_center/=2
            r1_patch_tmp=r1[ori_idx,r_center-patch_size/4:r_center+patch_size/4,c_center-patch_size/4:c_center+patch_size/4]
            mask=numpy.where(r1_patch_tmp<1)
            dmax = numpy.max(r1_patch_tmp[mask])
            dmin = numpy.min(r1_patch_tmp)
            r1_patch[new_idx][mask] = (r1_patch_tmp[mask] -dmin)/(dmax-dmin)

            r_center/=2
            c_center/=2
            r2_patch_tmp=r2[ori_idx,r_center-patch_size/8:r_center+patch_size/8,c_center-patch_size/8:c_center+patch_size/8]
            mask=numpy.where(r2_patch_tmp<1)
            dmax = numpy.max(r2_patch_tmp[mask])
            dmin = numpy.min(r2_patch_tmp)
            r2_patch[new_idx][mask] = (r2_patch_tmp[mask] -dmin)/(dmax-dmin)


    return r0_patch,r1_patch,r2_patch,new_prev_uvd_pred,new_cur_uvd_gr
def create_ori_patch(r0,r1,r2,prev_uvd_pred,prev_uvd_gr,hand_width,pad_width,patch_size):
    r0_patch= numpy.ones((r0.shape[0],patch_size,patch_size),dtype='float32')
    r1_patch= numpy.ones((r0.shape[0],patch_size/2,patch_size/2),dtype='float32')
    r2_patch= numpy.ones((r0.shape[0],patch_size/4,patch_size/4),dtype='float32')
    for ori_idx in xrange(r0.shape[0]):
            r_center = int(prev_uvd_pred[ori_idx,1]*hand_width+pad_width)
            c_center = int(prev_uvd_pred[ori_idx,0]*hand_width+pad_width)

            r0_patch_tmp=r0[ori_idx,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
            # plt.figure()
            # plt.imshow(r0[ori_idx],'gray')
            # plt.scatter(prev_uvd_pred[ori_idx,0]*hand_width+pad_width,prev_uvd_pred[ori_idx,1]*hand_width+pad_width,c='g')
            # plt.scatter(prev_uvd_gr[ori_idx,0]*hand_width+pad_width,prev_uvd_gr[ori_idx,1]*hand_width+pad_width,c='r')
            # plt.figure()
            # plt.imshow(r0_patch_tmp,'gray')
            # plt.show()
            mask=numpy.where(r0_patch_tmp<1)
            if mask[0].shape[0]==0:
                r_center = int(prev_uvd_gr[ori_idx,1]*hand_width+pad_width)
                c_center = int(prev_uvd_gr[ori_idx,0]*hand_width+pad_width)
                r0_patch_tmp=r0[ori_idx,r_center-patch_size/2:r_center+patch_size/2,c_center-patch_size/2:c_center+patch_size/2]
            mask=numpy.where(r0_patch_tmp<1)
            dmax = numpy.max(r0_patch_tmp[mask])
            dmin = numpy.min(r0_patch_tmp)

            r0_patch[ori_idx][mask] = (r0_patch_tmp[mask] -dmin)/(dmax - dmin)


            r_center/=2
            c_center/=2
            r1_patch_tmp=r1[ori_idx,r_center-patch_size/4:r_center+patch_size/4,c_center-patch_size/4:c_center+patch_size/4]
            mask=numpy.where(r1_patch_tmp<1)
            dmax = numpy.max(r1_patch_tmp[mask])
            dmin = numpy.min(r1_patch_tmp)
            r1_patch[ori_idx][mask] = (r1_patch_tmp[mask] -dmin)/(dmax-dmin)

            r_center/=2
            c_center/=2
            r2_patch_tmp=r2[ori_idx,r_center-patch_size/8:r_center+patch_size/8,c_center-patch_size/8:c_center+patch_size/8]
            mask=numpy.where(r2_patch_tmp<1)
            dmax = numpy.max(r2_patch_tmp[mask])
            dmin = numpy.min(r2_patch_tmp)
            r2_patch[ori_idx][mask] = (r2_patch_tmp[mask] -dmin)/(dmax-dmin)

    return r0_patch,r1_patch,r2_patch
def crop_patch_enlarge2(cur_uvd,prev_uvd_pred,prev_uvd_gr,r0,r1,r2,patch_size=24,patch_pad_width=4,hand_width=72,pad_width=12,batch_size=100):

    r0_patch,r1_patch,r2_patch= create_ori_patch(r0,r1,r2,prev_uvd_pred,prev_uvd_gr,hand_width,pad_width,patch_size)
    new_prev_uvd_pred = numpy.copy(prev_uvd_pred)
    new_cur_uvd_gr = numpy.copy(prev_uvd_gr)


    range_list = [[1,1.5],[1.5]]
    num_enlarge_list = [3,5]
    sum_num = 0
    for i in xrange(2):
        p0,p1,p2,pred_tmp,gr_tmp = create_replica2(ratio_range=range_list[i],num_enlarge=num_enlarge_list[i],
                                                                                    cur_uvd=cur_uvd,prev_uvd_pred=prev_uvd_pred,prev_uvd_gr=prev_uvd_gr,
                       r0=r0,r1=r1,r2=r2,hand_width=hand_width,pad_width=pad_width,patch_size=patch_size)
        r0_patch= numpy.concatenate([r0_patch,p0],axis=0)
        r1_patch= numpy.concatenate([r1_patch,p1],axis=0)
        r2_patch= numpy.concatenate([r2_patch,p2],axis=0)
        new_prev_uvd_pred= numpy.concatenate([new_prev_uvd_pred,pred_tmp],axis=0)
        new_cur_uvd_gr= numpy.concatenate([new_cur_uvd_gr,gr_tmp],axis=0)


    last_idx = (batch_size-sum_num%batch_size )
    rand_idx = numpy.random.randint(low=0,high=r0_patch.shape[0],size=last_idx)

    r0_patch= numpy.concatenate([r0_patch,r0_patch[rand_idx]],axis=0)
    r1_patch= numpy.concatenate([r1_patch,r1_patch[rand_idx]],axis=0)
    r2_patch= numpy.concatenate([r2_patch,r2_patch[rand_idx]],axis=0)
    new_prev_uvd_pred= numpy.concatenate([new_prev_uvd_pred,new_prev_uvd_pred[rand_idx]],axis=0)
    new_cur_uvd_gr= numpy.concatenate([new_cur_uvd_gr,new_cur_uvd_gr[rand_idx]],axis=0)


    border=patch_pad_width
    r0_patch = numpy.lib.pad(r0_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r1_patch = numpy.lib.pad(r1_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r2_patch = numpy.lib.pad(r2_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    # i=0
    # plt.figure()
    # plt.imshow(r0_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r1_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r2_patch[i],'gray')
    # plt.show()


    return r0_patch,r1_patch,r2_patch,new_cur_uvd_gr,new_prev_uvd_pred
def crop_patch_uvd_normalized_base(cur_uvd, prev_uvd,r0,r1,offset_depth_range=1.0,path_size=24,patch_pad_width=4,hand_width=72,pad_width=12):

    r0_patch= numpy.ones((r0.shape[0],path_size,path_size),dtype='float32')
    r1_patch= numpy.ones((r1.shape[0],path_size/2,path_size/2),dtype='float32')
    off_uvd = cur_uvd-prev_uvd
    off_uvd[:,0:2] = (off_uvd[:,0:2]*hand_width+path_size/2)/path_size
    print 'sample whose joint is beyonde the set offset depth range', numpy.where(numpy.abs(off_uvd[:,2])>offset_depth_range/2)[0].shape[0]

    for i in xrange(r0.shape[0]):
        r_center = int(prev_uvd[i,1]*hand_width+pad_width)
        c_center = int(prev_uvd[i,0]*hand_width+pad_width)
        r0_patch_tmp=r0[i,r_center-path_size/2:r_center+path_size/2,c_center-path_size/2:c_center+path_size/2]
        mask=numpy.where(r0_patch_tmp<1)
        dmax = numpy.max(r0_patch_tmp[mask])
        dmin = numpy.min(r0_patch_tmp)

        r0_patch[i][mask] = (r0_patch_tmp[mask] -dmin)/(dmax - dmin)
        off_uvd[i,2] = off_uvd[i,2]/offset_depth_range+0.5

        r_center/=2
        c_center/=2
        r1_patch_tmp=r1[i,r_center-path_size/4:r_center+path_size/4,c_center-path_size/4:c_center+path_size/4]
        mask=numpy.where(r1_patch_tmp<1)
        dmax = numpy.max(r1_patch_tmp[mask])
        dmin = numpy.min(r1_patch_tmp)
        r1_patch[i][mask] = (r1_patch_tmp[mask] -dmin)/(dmax-dmin)

        # plt.figure()
        # plt.imshow(r0_patch[i],'gray')
        # plt.scatter(off_uvd[i,0]*path_size,off_uvd[i,1]*path_size,c='r')
        # plt.scatter(path_size/2,path_size/2,c='g')
        # # plt.figure()
        # # plt.imshow(r0[i],'gray')
        # # plt.scatter(cur_uvd[i,0]*72+12,cur_uvd[i,1]*72+12,c='r')
        # plt.show()

    border=patch_pad_width
    r0_patch = numpy.lib.pad(r0_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r1_patch = numpy.lib.pad(r1_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    # i=0
    # plt.figure()
    # plt.imshow(r0_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r1_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r0[i],'gray')
    # plt.show()
    return r0_patch,r1_patch,off_uvd
def crop_patch_uvd_normalized_base_r0r1r2(cur_uvd, prev_uvd,r0,r1,r2,offset_depth_range=1.0,path_size=24,patch_pad_width=4,hand_width=72,pad_width=12):

    r0_patch= numpy.ones((r0.shape[0],path_size,path_size),dtype='float32')
    r1_patch= numpy.ones((r1.shape[0],path_size/2,path_size/2),dtype='float32')
    r2_patch= numpy.ones((r2.shape[0],path_size/4,path_size/4),dtype='float32')

    off_uvd = cur_uvd-prev_uvd.reshape((prev_uvd.shape[0],1,prev_uvd.shape[1]))
    if off_uvd.shape[1]>1:
        print 'sample whose joint is beyonde the set offset depth range', numpy.where(numpy.abs(off_uvd[:,:,2])>offset_depth_range/2)[0].shape[0]
        off_uvd[:,:,0:2] = (off_uvd[:,:,0:2]*hand_width+path_size/2)/path_size
        off_uvd[:,:,2] = off_uvd[:,:,2]/offset_depth_range+0.5

    else:
        print 'sample whose joint is beyonde the set offset depth range', numpy.where(numpy.abs(off_uvd[:,2])>offset_depth_range/2)[0].shape[0]
        off_uvd[:,0:2] = (off_uvd[:,0:2]*hand_width+path_size/2)/path_size
        off_uvd[:,2] = off_uvd[:,2]/offset_depth_range+0.5

    for i in xrange(r0.shape[0]):
        r_center = int(prev_uvd[i,1]*hand_width+pad_width)
        c_center = int(prev_uvd[i,0]*hand_width+pad_width)
        r0_patch_tmp=r0[i,r_center-path_size/2:r_center+path_size/2,c_center-path_size/2:c_center+path_size/2]
        mask=numpy.where(r0_patch_tmp<1)
        dmax = numpy.max(r0_patch_tmp[mask])
        dmin = numpy.min(r0_patch_tmp)

        r0_patch[i][mask] = (r0_patch_tmp[mask] -dmin)/(dmax - dmin)


        r_center/=2
        c_center/=2
        r1_patch_tmp=r1[i,r_center-path_size/4:r_center+path_size/4,c_center-path_size/4:c_center+path_size/4]
        mask=numpy.where(r1_patch_tmp<1)
        dmax = numpy.max(r1_patch_tmp[mask])
        dmin = numpy.min(r1_patch_tmp)
        r1_patch[i][mask] = (r1_patch_tmp[mask] -dmin)/(dmax-dmin)

        r_center/=2
        c_center/=2
        r2_patch_tmp=r2[i,r_center-path_size/8:r_center+path_size/8,c_center-path_size/8:c_center+path_size/8]
        mask=numpy.where(r2_patch_tmp<1)
        dmax = numpy.max(r2_patch_tmp[mask])
        dmin = numpy.min(r2_patch_tmp)
        r2_patch[i][mask] = (r2_patch_tmp[mask] -dmin)/(dmax-dmin)

        fig = plt.figure()
        ax= fig.add_subplot(131)
        ax.imshow(r0_patch[i],'gray')
        plt.scatter(off_uvd[i,:,0]*path_size,off_uvd[i,:,1]*path_size,c='g')
        plt.scatter(path_size/2,path_size/2,c='r')
        # plt.figure()
        # plt.imshow(r0[i],'gray')
        # plt.scatter(cur_uvd[i,0]*72+12,cur_uvd[i,1]*72+12,c='r')
        ax= fig.add_subplot(132)
        ax.imshow(r1_patch[i],'gray')
        ax= fig.add_subplot(133)
        ax.imshow(r2_patch[i],'gray')
        plt.show()

    border=patch_pad_width
    r0_patch = numpy.lib.pad(r0_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r1_patch = numpy.lib.pad(r1_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    border=patch_pad_width/2
    r2_patch = numpy.lib.pad(r2_patch, ((0,0),(border,border),(border,border)), 'constant',constant_values=1)
    # i=0
    # plt.figure()
    # plt.imshow(r0_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r1_patch[i],'gray')
    # plt.figure()
    # plt.imshow(r2_patch[i],'gray')
    # plt.show()
    return r0_patch,r1_patch,r2_patch,off_uvd

"""




