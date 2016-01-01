__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi
from src.Model.CNN_Model import CNN_Model_multi3_conv1
from src.Model.Train import update_params,get_gradients,update_params2,set_params
import time
from src.utils import constants
def train_model(num_iter,dataset,setname,dataset_path_prefix, source_name,batch_size,jnt_idx,c1,h1_out_factor,h2_out_factor,model_path,offset_save_path):

    model_info='uvd_bw_r012'
    print model_info
    if num_iter == 0:
        src_path = '%sdata/%s/hier_derot_recur/bw_initial/best/'%(dataset_path_prefix,setname)
    else:
        src_path = '%sdata/%s/hier_derot_recur/bw_offset/best/'%(dataset_path_prefix,setname)
    path = '%s%s%s.h5'%(src_path,dataset,source_name)

    test_set_x0, test_set_x1,test_set_x2,test_set_y=  load_data_multi(path=path,is_shuffle=False,
                                                                                             jnt_idx=jnt_idx)

    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    num_channel=test_set_x0.shape[1]
    n_test_batches = test_set_x0.shape[0]/ batch_size
    print 'n_test_batches', n_test_batches


    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    model = CNN_Model_multi3_conv1(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                             num_channel=num_channel,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 3,
                pool_c00= 2,
                c10= c1,
                kernel_c10= 3,
                pool_c10= 2,
                c20= c1,
                kernel_c20= 2,
                pool_c20= 1,

                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)

    cost = model.cost(Y)

    save_path = '%sdata/%s/hier_derot_recur/bw_offset/best/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_path)
    print model_save_path
    set_params(model_save_path, model.params)

    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')

    cost_nbatch = 0
    uvd_norm = numpy.empty_like(test_set_y)
    for minibatch_index in xrange(n_test_batches):
        slice_idx = range(minibatch_index * batch_size,(minibatch_index + 1) * batch_size,1)
        x0 = test_set_x0[slice_idx]
        x1 = test_set_x1[slice_idx]
        x2 = test_set_x2[slice_idx]
        y = test_set_y[slice_idx]

        cost_ij, uvd_batch = test_model(x0,x1,x2,numpy.cast['int32'](0), y)
        uvd_norm[slice_idx] = uvd_batch
        cost_nbatch+=cost_ij
    print 'cost', cost_nbatch/n_test_batches

    numpy.save("%s%s%s.npy"%(save_path,dataset,offset_save_path),uvd_norm)


if __name__ == '__main__':

    train_model(num_iter=0,
                dataset='test',
                dataset_path_prefix=constants.Data_Path,
                setname='msrc',
                source_name='_iter0_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm0_lm2000_yt0_ep1500',
                model_path='param_cost_uvd_bw5_r012_egoff2_c0064_h11_h22_gm0_lm3000_yt0_ep185',
                offset_save_path='_uvd_bw5_r012_egoff2_c0064_h11_h22_gm0_lm3000_yt0_ep185',
                batch_size = 100,
                jnt_idx = [5],
                c1=64,
                h1_out_factor=1,
                h2_out_factor=2)
    #
    # train_model(dataset='test',
    #             dataset_path_prefix=dataset_path_prefix,
    #             setname='nyu',
    #             source_name='_iter1_patch_derot_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm9900_lm1038_yt0_ep2020',
    #             model_path='param_cost_uvd_bw1_r012_egoff2_c0064_h11_h22_gm0_lm9999_yt0_ep545',
    #             offset_save_path='_uvd_bw1_r012_egoff2_c0064_h11_h22_gm0_lm9999_yt0_ep545',
    #             batch_size = 100,
    #             jnt_idx = [1],
    #             c1=64,
    #             h1_out_factor=1,
    #             h2_out_factor=2)
    #


    # train_model(num_iter=0,
    #             dataset='test',
    #             batch_size = 133,
    #             setname='icvl',
    #             dataset_path_prefix=dataset_path_prefix,
    #             source_name='_iter0_patch_uvd_bw_r012_21jnts_c0016_c0132_c1016_c1132_c2016_c2132_h12_h24_gm5051_lm2000_yt0_ep2450',
    #             model_path='param_cost_uvd_bw0_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep465',
    #             offset_save_path='_uvd_bw0_r012_egoff2_c0064_h12_h22_gm0_lm1000_yt0_ep465',
    #
    #             jnt_idx = [0],
    #             c1=64,
    #             h1_out_factor=1,
    #             h2_out_factor=2)
