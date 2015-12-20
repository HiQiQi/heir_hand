from src.utils import constants

__author__ = 'QiYE'
import numpy
from PIL import Image
import scipy.io
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation  as interplt
import h5py


def segment_hand_image_one(depth, jnt_uvd):
    umin = numpy.min(jnt_uvd[:, 0])
    umax = numpy.max(jnt_uvd[:, 0])

    vmin = numpy.min(jnt_uvd[:, 1])
    vmax = numpy.max(jnt_uvd[:, 1])
    margin = 35
    diff = numpy.abs(umax - umin - (vmax - vmin))
    mask_u = (umax - umin) < (vmax - vmin)
    mask_v = ~mask_u
    mask_u = numpy.array(mask_u, dtype='uint16')
    mask_v = numpy.array(mask_v, dtype='uint16')
    umin = int(numpy.floor(umin - mask_u * diff / 2 - margin))
    umax = int(numpy.floor(umax + mask_u * diff / 2 + margin))
    vmin = int(numpy.floor(vmin - mask_v * diff / 2 - margin))
    vmax = int(numpy.floor(vmax + mask_v * diff / 2 + margin))

    hand_image = depth[vmin:vmax, umin:umax]
    #pylab.imshow(hand_image, cmap='gray', norm=colors.Normalize(vmin=0, vmax=numpy.max(depth)))
    # pylab.show()

    mask = (hand_image >= (numpy.min(jnt_uvd[:, 2]) - 20)) & (hand_image <= (numpy.max(jnt_uvd[:, 2]) + 20))
    hand_image = numpy.asarray(hand_image, dtype='float32')
    lochand = numpy.where(mask)
    hand_image /= numpy.mean(hand_image[lochand])
    dmin = numpy.min(hand_image[lochand])
    dmax = numpy.max(hand_image[lochand])
    hand_image = (hand_image - dmin) / (dmax - dmin)
    loc = numpy.where(~mask)
    hand_image[loc] = 1
    size = 96,96
    handpixel = Image.fromarray(hand_image)
    handpixel = handpixel.resize(size, Image.ANTIALIAS)
    #handpixel.show()
    hand_image = numpy.array(handpixel, dtype='float32')
    #pylab.imshow(hand_image, cmap='gray', norm=colors.Normalize(vmin=0, vmax=1))
    #pylab.show()
    return hand_image
def creat_jnt_label(jnt_uvd,  umin, vmin, range):
    ratio = range / 18.0
    new_jnt_uvd = numpy.empty((jnt_uvd.shape[0], jnt_uvd.shape[1]),dtype='uint8')
    v_offset = vmin
    u_offst = umin

    new_jnt_uvd[:, 1] = numpy.asarray(numpy.floor((jnt_uvd[:, 1] - v_offset)/ratio), dtype='uint8')
    new_jnt_uvd[:, 0] = numpy.asarray(numpy.floor((jnt_uvd[:, 0] - u_offst)/ratio), dtype='uint8')
    new_jnt_uvd[:, 2] = 1#jnt_uvd[:, 2]
    #print new_jnt_uvd
    jnt_image = numpy.zeros(( constants.NUM_JNTS, 18, 18), dtype='float32')
    #image = numpy.zeros((18, 18), dtype='float32')
    """
    the order in jnt label is u, v, d
    the corresponding array for an image is v*u, v is row index axis = 0, u is column index, axis = 1
    """
    for i in numpy.arange(0,  constants.NUM_JNTS, 1):
        jnt_image[i, new_jnt_uvd[i, 1], new_jnt_uvd[i, 0]] = new_jnt_uvd[i, 2]

    jnt_image.shape = (1,  constants.NUM_JNTS*18*18)
    # return new_jnt_uvd, jnt_image
    return  jnt_image
   # jnt_lables = numpy.array((depth.shape[0], 4536), dtype='float32')



def load_segment_hand_image_all_msrc(dataset_dir,
                                fram_prefix,
                                num_img,
                                jnt_uvd,
                                img_size = 96,
                                w = 72,
                                margin = 25,
                                orig_pad_border = 25):
    """

    return  hand_arrays, jnt_labels, rect_d1d2w

    hand_arrays:
    the size of the normalized hand image is (img_size,img_size)
    with the hand in the center whose size  (w,w)
    and the width of padding around the hand (img_size - w)/2

    jnt_labels:
    the location of joints in the normalized hand image  (img_size,img_size)

    rect_d1d2w:
    the bbox of the hand area in the original image,
    d0-->v
    d1-->u
    w-->width


    other notes:
    u ; image: width, column; array, dimension 1
    v ; image: height, row; dimension 0
    hand_array = depth[ vmin:(vmin+urange), umin:(umin+urange)]
    -------------u-----------------
    |
    |
    |
    v
    |
    |
    |

    """

    pad_width = (img_size - w)/2
    hand_arrays = numpy.empty((num_img, w, w),dtype='float32')
    jnt_uvd_norm = numpy.empty((num_img, constants.NUM_JNTS,3), dtype='float32')
    depth_dmin_dmax = numpy.empty((num_img,2),dtype='float32')
    handpixel = numpy.empty((w,w),dtype='float32')

    rect_d1d2w = numpy.empty((num_img, 3), dtype='float32')

    """some hand area is closed to the image boundary.
    Add some blank border so that the hand area extraced won't go beyond the original image boundary"""
    print jnt_uvd[0,0,:]
    jnt_uvd[:,:,0] += 25
    jnt_uvd[:,:,1] += 25
    low =0
    high=0
    for i in range(0,num_img,1):
        # print i
        filename_prefix = "%06d" % (i)
        depth = Image.open('%s%s_%s_depth.png' % (dataset_dir, fram_prefix, filename_prefix))
        depth = numpy.asarray(depth, dtype='uint16')

        depth = numpy.lib.pad(depth, ((orig_pad_border, orig_pad_border), (orig_pad_border, orig_pad_border)), 'constant', constant_values=5000)
        """d1 d2 are the left top axes of the hand area to be extracted"""
        d1 = depth.shape[0]
        d2 = depth.shape[1]
        # plt.imshow(depth[:, :], cmap='gray')
        # plt.scatter(jnt_uvd[i, :, 0], jnt_uvd[i,:, 1], s=20, c='g')
        # plt.show()
        # print jnt_uvd[i]
        umin = numpy.min(jnt_uvd[i, :, 0])
        umax = numpy.max(jnt_uvd[i, :, 0])
        vmin = numpy.min(jnt_uvd[i, :, 1])
        vmax = numpy.max(jnt_uvd[i, :, 1])


        diff = numpy.abs(umax - umin - (vmax - vmin))
        #print umax - umin, '  ', vmax - vmin
        mask_u = (umax - umin) < (vmax - vmin)
        mask_v = ~mask_u
        mask_u = numpy.asarray(mask_u, dtype='uint16')
        mask_v = numpy.asarray(mask_v, dtype='uint16')
        umin = numpy.asarray((numpy.floor(umin - mask_u * diff / 2 - margin)), dtype='float32')
        umax = numpy.array(numpy.floor(umax + mask_u * diff / 2 + margin), dtype='float32')
        vmin = numpy.asarray(numpy.floor(vmin - mask_v * diff / 2 - margin), dtype='float32')
        urange = umax - umin
        if vmin < 0:
            vmin = 0
        if vmin+urange > d1:
            vmin = d1 - urange
        if umin < 0:
             umin = 0
        if umin+urange > d2:
            umin = d2-urange
        rect_d1d2w[i] =( vmin,umin,urange)
        hand_array = depth[ vmin:(vmin+urange), umin:(umin+urange)]
        # plt.figure()
        # plt.imshow(hand_array,'gray')
        # plt.show()
        """the snippet if for real depth image"""
        # min_depth = numpy.min(jnt_uvd[i, :, 2]) - depth_range
        # max_depth = numpy.max(jnt_uvd[i, :, 2]) + depth_range
        # mask = numpy.logical_and((hand_array >= min_depth), (hand_array <= max_depth))
        # hand_array = numpy.asarray(hand_array, dtype='float32')
        # lochand = numpy.where(mask)

        # dmin = numpy.min(hand_array[lochand])
        # dmax = numpy.max(hand_array[lochand])
        """"""

        # center_hand = numpy.asarray([numpy.mean(lochand[0]), numpy.mean(lochand[1])], dtype='int32')
        # mean_depth = 0
        # for ic in xrange(center_hand[0]-2,center_hand[0]+3):
        #     for jc in xrange( center_hand[1]-2, center_hand[1]+3):
        #         mean_depth += hand_array[ic,jc]
        # mean_depth /= 25
        # # print mean_depth
        # if mean_depth <= min_depth or mean_depth >=max_depth:
        #     mean_depth = numpy.mean(hand_array[lochand])
        # mean_depth = numpy.mean(hand_array[lochand])
        # hand_array -= mean_depth

        """no mask for the synthetic, for outsize the handarea, all the pixels are 50000"""
        depth_mean = numpy.mean(jnt_uvd[i, :, 2])
        dmin=depth_mean-150
        dmax=depth_mean+150
        mask = numpy.logical_and((hand_array >dmin), (hand_array <dmax))
        hand_array = numpy.asarray(hand_array, dtype='float32')

        depth_dmin_dmax[i]=(dmin,dmax)
        hand_array = (hand_array - dmin) / (dmax - dmin)#normalize the depth value to 0 and 1
        loc = numpy.where(~mask)
        hand_array[loc] = 1

        jnt_uvd_norm[i, :, 2] = (jnt_uvd[i, :, 2]- dmin) / (dmax - dmin)
        # print dmin,dmax, jnt_uvd[i, :, 2],jnt_uvd_norm[i, :, 2]
        # print dmax-dmin,numpy.max(jnt_uvd[i, :, 2])-numpy.min(jnt_uvd[i, :, 2])
        interplt.zoom(hand_array, w/urange, handpixel,order=1, mode='reflect',prefilter=True)
        hand_arrays[i, :, :] =handpixel
        # plt.figure()
        # plt.imshow(handpixel,'gray')
        # plt.show()
        # ratio = urange/w
        # for jnt_idx in xrange(constants.NUM_JNTS):
        #     cir = plt.Circle(((jnt_uvd[i, jnt_idx, 0]-umin)/ratio,(jnt_uvd[i, jnt_idx, 1]-vmin)/ratio),1,color='r')
        #     fig.gca().add_artist(cir)

        # handpixel = Image.fromarray(hand_array)
        # handpixel = handpixel.resize((w, w), Image.NEAREST)
        # hand_arrays[i , :, :] = numpy.array(handpixel, dtype='float32')
        # plt.figure()
        # plt.imshow(hand_arrays[i, :, :],'gray')
        #
        # print 'image',i
        jnt_uvd_norm[i, :, 0] = (jnt_uvd[i, :, 0] - umin )/urange
        jnt_uvd_norm[i, :, 1] = (jnt_uvd[i, :, 1] - vmin )/urange

        # print new_jnt_uvd[:,0]
        # print new_jnt_uvd[:, 0]*urange/18.0
        # print numpy.ceil(new_jnt_uvd[:, 0]*urange/18.0)
        # cc = urange/18/2
        # plt.scatter((new_jnt_uvd[:, 0]*urange/18.0+cc),((new_jnt_uvd[:, 1])*urange/18.0+cc), s=20, c='r')
        # plt.show()
    # print 'err_num',err_num
    border = int(pad_width)
    hand_arrays = numpy.lib.pad(hand_arrays, ((0, 0), (border, border), (border, border)), 'constant', constant_values=1)
    # plt.figure()
    # plt.imshow(hand_arrays[0, :, :],'gray')
    # plt.show()
    print numpy.max(jnt_uvd_norm[:,:,2])
    print numpy.min(jnt_uvd_norm[:,:,2])
    print numpy.where(jnt_uvd_norm[:,:,2]>1)[0].shape[0]
    print numpy.where(jnt_uvd_norm[:,:,2]>1.1)[0].shape[0]
    print numpy.where(jnt_uvd_norm[:,:,2]<0)[0].shape[0]
    print low,high
    return hand_arrays, jnt_uvd_norm, rect_d1d2w, depth_dmin_dmax


def load_segment_hand_image_all(dataset_dir,
                                fram_prefix,
                                fram_pstfix,
                                num_img,
                                jnt_uvd,
                                img_size = 96,
                                w = 72,
                                margin = 25,
                                orig_pad_border = 25,
                                depth_range = 200):
    """

    return  hand_arrays, jnt_labels, rect_d1d2w

    hand_arrays:
    the size of the normalized hand image is (img_size,img_size)
    with the hand in the center whose size  (w,w)
    and the width of padding around the hand (img_size - w)/2

    jnt_labels:
    the location of joints in the normalized hand image  (img_size,img_size)

    rect_d1d2w:
    the bbox of the hand area in the original image,
    d0-->v
    d1-->u
    w-->width


    other notes:
    u ; image: width, column; array, dimension 1
    v ; image: height, row; dimension 0
    hand_array = depth[ vmin:(vmin+urange), umin:(umin+urange)]
    -------------u-----------------
    |
    |
    |
    v
    |
    |
    |

    """

    pad_width = (img_size - w)/2
    hand_arrays = numpy.empty((num_img, w, w),dtype='float32')
    jnt_uvd_norm = numpy.empty((num_img, constants.NUM_JNTS,3), dtype='float32')
    depth_dmin_dmax = numpy.empty((num_img,2),dtype='float32')
    handpixel = numpy.empty((w,w),dtype='float32')

    rect_d1d2w = numpy.empty((num_img, 3), dtype='float32')

    """some hand area is closed to the image boundary.
    Add some blank border so that the hand area extraced won't go beyond the original image boundary"""
    print jnt_uvd[0,0,:]
    jnt_uvd[:,:,0] += 25
    jnt_uvd[:,:,1] += 25

    for i in range(0,num_img,1):
        # print i
        filename_prefix = "%07d" % (i+1)
        depth = Image.open('%s%s_%s%s.png' % (dataset_dir, fram_prefix, filename_prefix,fram_pstfix))
        depth = numpy.asarray(depth, dtype='uint16')
        # depth = numpy.asarray(depth, dtype='uint16')
        # depth = depth[:, :, 2]+numpy.left_shift(depth[:, :, 1], 8)


        depth = numpy.lib.pad(depth, ((orig_pad_border, orig_pad_border), (orig_pad_border, orig_pad_border)), 'constant', constant_values=5000)
        """d1 d2 are the left top axes of the hand area to be extracted"""
        d1 = depth.shape[0]
        d2 = depth.shape[1]
        # plt.imshow(depth[:, :], cmap='gray')
        # plt.scatter(jnt_uvd[i, :, 0], jnt_uvd[i,:, 1], s=20, c='g')
        # plt.show()
        # print jnt_uvd[i]
        umin = numpy.min(jnt_uvd[i, :, 0])
        umax = numpy.max(jnt_uvd[i, :, 0])
        vmin = numpy.min(jnt_uvd[i, :, 1])
        vmax = numpy.max(jnt_uvd[i, :, 1])


        diff = numpy.abs(umax - umin - (vmax - vmin))
        #print umax - umin, '  ', vmax - vmin
        mask_u = (umax - umin) < (vmax - vmin)
        mask_v = ~mask_u
        mask_u = numpy.asarray(mask_u, dtype='uint16')
        mask_v = numpy.asarray(mask_v, dtype='uint16')
        umin = numpy.asarray((numpy.floor(umin - mask_u * diff / 2 - margin)), dtype='float32')
        umax = numpy.array(numpy.floor(umax + mask_u * diff / 2 + margin), dtype='float32')
        vmin = numpy.asarray(numpy.floor(vmin - mask_v * diff / 2 - margin), dtype='float32')
        urange = umax - umin
        if vmin < 0:
            vmin = 0
        if vmin+urange > d1:
            vmin = d1 - urange
        if umin < 0:
             umin = 0
        if umin+urange > d2:
            umin = d2-urange
        rect_d1d2w[i] =( vmin,umin,urange)
        hand_array = depth[ vmin:(vmin+urange), umin:(umin+urange)]
        # plt.figure()
        # plt.imshow(hand_array,'gray')
        # plt.show()

        """no mask for the synthetic, for outsize the handarea, all the pixels are 50000"""
        depth_mean = numpy.mean(jnt_uvd[i, :, 2])
        dmin=depth_mean-depth_range/2
        dmax=depth_mean+depth_range/2

        mask = numpy.logical_and((hand_array >dmin), (hand_array <dmax))
        hand_array = numpy.asarray(hand_array, dtype='float32')

        depth_dmin_dmax[i]=(dmin,dmax)
        hand_array = (hand_array - dmin) / (dmax - dmin)#normalize the depth value to 0 and 1
        loc = numpy.where(~mask)
        hand_array[loc] = 1

        jnt_uvd_norm[i, :, 2] = (jnt_uvd[i, :, 2]- dmin) / (dmax - dmin)
        # print dmin,dmax, jnt_uvd[i, :, 2],jnt_uvd_norm[i, :, 2]
        # print dmax-dmin,numpy.max(jnt_uvd[i, :, 2])-numpy.min(jnt_uvd[i, :, 2])
        interplt.zoom(hand_array, w/urange, handpixel,order=1, mode='reflect',prefilter=True)
        hand_arrays[i, :, :] =handpixel
        jnt_uvd_norm[i, :, 0] = (jnt_uvd[i, :, 0] - umin )/urange
        jnt_uvd_norm[i, :, 1] = (jnt_uvd[i, :, 1] - vmin )/urange
        # plt.figure()
        # plt.imshow(handpixel,'gray')
        # plt.scatter(jnt_uvd_norm[i,:,0]*72,jnt_uvd_norm[i,:,1]*72)
        # plt.show()


    # print 'err_num',err_num
    border = int(pad_width)
    hand_arrays = numpy.lib.pad(hand_arrays, ((0, 0), (border, border), (border, border)), 'constant', constant_values=1)
    plt.figure()
    plt.imshow(hand_arrays[0, :, :],'gray')
    plt.scatter(jnt_uvd_norm[0,:,0]*72+border,jnt_uvd_norm[0,:,1]*72+border)
    plt.show()
    print numpy.max(jnt_uvd_norm[:,:,2])
    print numpy.min(jnt_uvd_norm[:,:,2])
    print numpy.max(jnt_uvd_norm[:,:,1])
    print numpy.min(jnt_uvd_norm[:,:,1])
    print numpy.max(jnt_uvd_norm[:,:,0])
    print numpy.min(jnt_uvd_norm[:,:,0])
    return hand_arrays, jnt_uvd_norm, rect_d1d2w, depth_dmin_dmax

def multi_resolution_image_zoom(depth):

    num_img = depth.shape[0]
    w = depth.shape[1]
    r1 = numpy.empty((num_img,w/2, w/2), dtype='float32')
    r2 = numpy.empty((num_img, w/4, w/4), dtype='float32')
    for i in xrange(num_img):
        r1[i] = interplt.zoom(depth[i], 0.5, order=1, mode='reflect',prefilter=True)
        r2[i] = interplt.zoom(depth[i], 0.25, order=1, mode='reflect',prefilter=True)
    return r1, r2


def hand_image_pyramid_msrc():
    """
    The function is to extract hand area of samples in the msrc dataset


    """
    # data_set = 'train'
    # fram_prefix = 'random'
    data_set = 'test'
    fram_prefix = 'testds'
    dataset_dir = 'C:\\Proj\\msrc_data\\2014-05-19 personalizations global\\%s\\'%data_set
    save_path = '../../data/'
    img_size=96
    norm_center_hand_width = 72
    margin_around_center_hand=25

    orig_pad_border=25
    print '%s'%dataset_dir

    keypoints = scipy.io.loadmat('../../data/joint_uvd/uvd_msrc_%s_22joints.mat' % data_set)
    joint_uvd = keypoints['uvd'][:,1:22,:]
    num_img = joint_uvd.shape[0]
    print 'num_img',num_img

    hand_arrays, joint_label, rect,depth_dmin_dmax =  load_segment_hand_image_all_msrc(dataset_dir=dataset_dir,
                                                                  fram_prefix=fram_prefix,
                                                                  num_img=num_img,
                                                                  img_size=img_size,
                                                                  jnt_uvd=joint_uvd,
                                                                    w = norm_center_hand_width,
                                                                    margin = margin_around_center_hand,
                                                                    orig_pad_border = orig_pad_border)

    r1, r2 = multi_resolution_image_zoom(hand_arrays)


    save_path_name = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300.h5'%(save_path,data_set)
    f = h5py.File(save_path_name, 'w')
    f.create_dataset('r0', data=hand_arrays)
    f.create_dataset('r1', data=r1)
    f.create_dataset('r2', data=r2)
    f.create_dataset('joint_label_uvd', data=joint_label)
    f.create_dataset('bbox', data=rect)
    f.create_dataset('depth_dmin_dmax', data=depth_dmin_dmax)
    f.create_dataset('norm_center_hand_width', data=norm_center_hand_width)
    f.create_dataset('margin_around_center_hand', data=margin_around_center_hand)
    f.create_dataset('orig_pad_border', data=orig_pad_border)
    f.close()


    print 'file closed.'
def hand_image_pyramid(data_name,data_set,fram_prefix,fram_pstfix,dataset_dir,save_path,img_size,norm_center_hand_width,margin_around_center_hand,orig_pad_border,depth_range):
    """
    The function is to extract hand area of samples in the msrc dataset
    """
    print '%s'%dataset_dir

    keypoints = scipy.io.loadmat('../../data/%s/source/%s_%s_uvd_21joints.mat' % (data_name,data_set,data_name))
    joint_uvd = keypoints['uvd']
    num_img = joint_uvd.shape[0]
    print 'num_img',num_img

    hand_arrays, joint_label, rect,depth_dmin_dmax =  load_segment_hand_image_all(dataset_dir=dataset_dir,
                                                                  fram_prefix=fram_prefix,
                                                                  fram_pstfix=fram_pstfix,
                                                                  num_img=num_img,
                                                                  img_size=img_size,
                                                                  jnt_uvd=joint_uvd,
                                                                    w = norm_center_hand_width,
                                                                    margin = margin_around_center_hand,
                                                                    orig_pad_border = orig_pad_border,
                                                                    depth_range=depth_range)

    r1, r2 = multi_resolution_image_zoom(hand_arrays)


    save_path_name = '%s%s_%s_r0_r1_r2_uvd_bbox_21jnts_20151125_depth%s.h5'%(save_path,data_set,data_name,depth_range)
    f = h5py.File(save_path_name, 'w')
    f.create_dataset('r0', data=hand_arrays)
    f.create_dataset('r1', data=r1)
    f.create_dataset('r2', data=r2)
    f.create_dataset('joint_label_uvd', data=joint_label)
    f.create_dataset('bbox', data=rect)
    f.create_dataset('depth_dmin_dmax', data=depth_dmin_dmax)
    f.create_dataset('norm_center_hand_width', data=norm_center_hand_width)
    f.create_dataset('margin_around_center_hand', data=margin_around_center_hand)
    f.create_dataset('orig_pad_border', data=orig_pad_border)
    f.close()


    print 'file closed.'
if __name__ == '__main__':
    data_set = 'train'
    data_name='icvl'
    hand_image_pyramid(data_name=data_name,
                            data_set =data_set,
    fram_prefix = 'depth_1',
    fram_pstfix = '',
    dataset_dir = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\ICVL_dataset_v2_msrc_format\\%s\\depth\\'%data_set,
    save_path = '../../data/icvl/source/',
    img_size=96,
    norm_center_hand_width = 72,
    margin_around_center_hand=25,
    orig_pad_border=25,
    depth_range=200)

    # data_set = 'train'
    # data_name='nyu'
    # fram_prefix='depth_1'
    # hand_image_pyramid(data_name=data_name,
    #                         data_set =data_set,
    # fram_prefix = fram_prefix,
    # fram_pstfix = '',
    # dataset_dir = 'D:\\Project\\3DHandPose\\Data_3DHandPoseDataset\\NYU\\NYU_dataset\\%s\\'%data_set,
    # save_path = '../../data/nyu/source/',
    # img_size=96,
    # norm_center_hand_width = 72,
    # margin_around_center_hand=25,
    # orig_pad_border=25,
    # depth_range=300)