import h5py
import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3,CNN_Model_multi3_conv3
from src.Model.Train import update_params,get_gradients,update_params2,set_params
from src.utils.crop_patch_norm_offset import crop_bw_ego_conv_patch
from src.utils.rotation import recur_derot
import time
from src.utils import constants


import matplotlib.pyplot as plt
def show_img_loc(img,pred_uvd,gr_uvd):
    img_size=img.shape[-1]
    for i in numpy.random.randint(0,img.shape[0],10):
        pred_uvd_tmp = pred_uvd[i,:,0:2]*img_size
        gr_uvd_tmp =gr_uvd[i,:,0:2]*img_size
        plt.imshow(img[i,0],'gray')
        plt.scatter(pred_uvd_tmp[:,0],pred_uvd_tmp[:,1],c='r')
        plt.scatter(gr_uvd_tmp[:,0],gr_uvd_tmp[:,1],c='g')
        plt.show()


def crop_conv3(patch_size,dataset_path_prefix,dataset,setname, source_name,batch_size,jnt_idx,
                     c1,c2,c3,h1_out_factor,h2_out_factor,model_path,offset_save_path):

    model_info='uvd_bw_r012_21jnts'
    print model_info
    src_path = '%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi(path,is_shuffle=False, jnt_idx=jnt_idx)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')

    model = CNN_Model_multi3_conv3(
        model_info=model_info,
        X0=X0,X1=X1,X2=X2,
              img_size_0 = img_size_0,
              img_size_1=img_size_1,
              img_size_2=img_size_2,
              is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 4,
                pool_c01= 2 ,
                c02= c3,
                kernel_c02= 3,
                pool_c02= 2,

                c10= c1,
                kernel_c10= 5,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 3,
                pool_c11= 2 ,
                c12= c3,
                kernel_c12= 3,
                pool_c12= 2 ,

                c20= c1,
                kernel_c20= 5,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 5,
                pool_c21= 1 ,
                c22= c3,
                kernel_c22= 3,
                pool_c22= 1 ,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)


    cost = model.cost(Y)

    save_path = '%sdata/%s/hier_derot_recur/bw_initial/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
    print model_save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[0].output,model.layers[3].output,model.layers[6].output,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    pred_uvd_derot = numpy.empty_like(test_set_y)
    gr_uvd_derot = numpy.empty_like(test_set_y)
    pred_uvd = numpy.empty_like(test_set_y)

    patch00 = numpy.empty((6,test_set_y.shape[0],c1,patch_size,patch_size),dtype='float32')
    patch10 = numpy.empty((6,test_set_y.shape[0],c1,patch_size,patch_size),dtype='float32')
    patch20 = numpy.empty((6,test_set_y.shape[0],c1,patch_size/2,patch_size/2),dtype='float32')
    rotation = numpy.empty((test_set_y.shape[0],),dtype='float32')
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, c00_out,c10_out,c20_out,pred_uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)

        rot_batch,pred_uvd_derot_batch, gr_uvd_derot_batch,pred_patch_center,gr_patch_center = recur_derot(c00_out,
                                                               c10_out,c20_out,pred_uvd=pred_uvd_batch.reshape(batch_size,6,3),gr_uvd=y.reshape(batch_size,6,3),batch_size=batch_size)

        # show_img_loc(c00_out,pred_patch_center[0],gr_patch_center[0])
        # show_img_loc(c20_out,pred_patch_center[2],gr_patch_center[2])
        # show_img_loc(c10_out,pred_patch_center[1],gr_patch_center[1])


        patch00_tmp, patch10_tmp, pathc20_tmp = crop_bw_ego_conv_patch(c00_out,c10_out,c20_out,pred_patch_center,patch_size=8)

        pred_uvd[slice_idx] = pred_uvd_batch
        rotation[slice_idx]=rot_batch
        patch00[:,slice_idx,:,:,:] = patch00_tmp
        patch10[:,slice_idx,:,:,:] = patch10_tmp
        patch20[:,slice_idx,:,:,:] = pathc20_tmp

        pred_uvd_derot[slice_idx] = pred_uvd_derot_batch.reshape(batch_size,18)
        gr_uvd_derot[slice_idx] = gr_uvd_derot_batch.reshape(batch_size,18)
        cost_nbatch+=cost_ij

    # for i in numpy.random.randint(0,patch10.shape[0],10):
    #     fig=plt.figure()
    #     ax=fig.add_subplot(131)
    #     ax.imshow(patch00[0,i,0],'gray')
    #
    #     ax=fig.add_subplot(132)
    #     ax.imshow(patch10[0,i,0],'gray')
    #
    #     ax=fig.add_subplot(133)
    #     ax.imshow(patch20[0,i,0],'gray')
    #     plt.show()
    offset = gr_uvd_derot-pred_uvd_derot
    loc = numpy.where(numpy.max(offset))
    print loc
    print offset[loc]
    print pred_uvd_derot[loc]
    print gr_uvd_derot[loc]
    print test_set_y[loc]

    loc = numpy.where(numpy.min(offset))
    print loc
    print offset[loc]
    print pred_uvd_derot[loc]
    print gr_uvd_derot[loc]
    print test_set_y[loc]

    print 'cost', cost_nbatch/n_test_batches
    save_path = '%sdata/%s/hier_derot_recur/bw_initial/best/'%(dataset_path_prefix,setname)
    path = "%s%s%s.h5"%(save_path,dataset,offset_save_path)
    f_shf = h5py.File(path,'w')
    f_shf.create_dataset('patch00',data=patch00)
    f_shf.create_dataset('patch10',data=patch10)
    f_shf.create_dataset('patch20',data=patch20)
    f_shf.create_dataset('pred_uvd',data=pred_uvd)
    f_shf.create_dataset('gr_uvd',data=test_set_y)
    f_shf.create_dataset('pred_uvd_derot',data=pred_uvd_derot)
    f_shf.create_dataset('gr_uvd_derot',data=gr_uvd_derot)
    f_shf.create_dataset('rotation',data=rotation)
    f_shf.close()



if __name__ == '__main__':

    # crop_conv3(patch_size=8,
    #            dataset='train',
    #                  setname='msrc',
    #         source_name='_msrc_r0_r1_r2_uvd_bbox_21jnts_20151030_depth300',
    #             batch_size =100 ,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=64,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_recur1_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500',
    #             model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500')

    #
    # crop_conv3(patch_size=8,
    #            dataset='test',
    #                  setname='nyu',
    #         source_name='_nyu_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             batch_size =8 ,
    #             jnt_idx = [0,1,5,9 ,13,17],
    #             c1=16,
    #             c2=32,
    #             c3=48,
    #             h1_out_factor=2,
    #             h2_out_factor=4,
    #             offset_save_path='_recur1_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020',
    #             model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020')

    dataset_path_prefix=constants.Data_Path
    crop_conv3(patch_size=8,
               dataset_path_prefix=dataset_path_prefix,
               dataset='train',
                     setname='icvl',
            source_name='_icvl_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                batch_size =8 ,
                jnt_idx = [0,1,5,9 ,13,17],
                c1=16,
                c2=32,
                c3=64,
                h1_out_factor=2,
                h2_out_factor=4,
                offset_save_path='_iter0_patch_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm5051_lm2000_yt0_ep2450',
                model_path='param_cost_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm5051_lm2000_yt0_ep2450')



