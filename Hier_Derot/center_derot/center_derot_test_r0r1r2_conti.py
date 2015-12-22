from src.utils import constants

__author__ = 'QiYE'

import theano
import theano.tensor as T
import numpy
from load_data import  load_data_multi_center_continuous
from src.Model.CNN_Model import CNN_Model_multi3
from src.Model.Train import set_params
def train_model_sequence(setname,dataset_path_prefix,dataset, source_name,batch_size,model_save_path,pred_save_name):

    model_info='center_r0r1r2_uvd'
    print model_info, constants.OUT_DIM
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    # path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot.h5'%(src_path,dataset)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_center_continuous(path,is_shuffle=False)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=6
    h2_out_factor=16
    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 3,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= c1,
                kernel_c20= 3,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)
    cost = model.cost(Y)


    test_model = theano.function(inputs=[X0,X1,X2,is_train,Y],
        outputs=[cost,model.layers[-1].output], on_unused_input='ignore')
    for i in xrange(1930,2050,5):

        save_path = '%sdata/%s/hier_derot/center/'%(dataset_path_prefix,setname)
        model_save_path = "%sparam_cost_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep%d.npy"%(save_path,i)
        set_params(model_save_path, model.params)

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
        print 'ep',i, ' cost', cost_nbatch/n_test_batches


def train_model(setname,dataset_path_prefix,dataset, source_name,batch_size,model_save_path,pred_save_name):

    model_info='center_r0r1r2_uvd'
    print model_info, constants.OUT_DIM
    src_path ='%sdata/%s/source/'%(dataset_path_prefix,setname)

    path = '%s%s%s.h5'%(src_path,dataset,source_name)
    # path = '%smsrc_%s_r0_r1_r2_uvd_bbox_21jnts_derot.h5'%(src_path,dataset)
    test_set_x0, test_set_x1,test_set_x2,test_set_y= load_data_multi_center_continuous(path,is_shuffle=False)
    n_test_batches = test_set_x0.shape[0]/ batch_size
    img_size_0 = test_set_x0.shape[2]
    img_size_1 = test_set_x1.shape[2]
    img_size_2 = test_set_x2.shape[2]
    print 'n_test_batches', n_test_batches

    X0 = T.tensor4('source0')   # the data is presented as rasterized images
    X1 = T.tensor4('source1')
    X2 = T.tensor4('source2')
    is_train =  T.iscalar('is_train')
    # x0.tag.test_value = train_set_x0.get_value()
    Y = T.matrix('target')
    c1=16
    c2=32
    h1_out_factor=6
    h2_out_factor=16
    model = CNN_Model_multi3(X0=X0,X1=X1,X2=X2,
                             model_info=model_info,
                      img_size_0 = img_size_0,
                      img_size_1=img_size_1,
                      img_size_2=img_size_2,
                      is_train=is_train,
                c00= c1,
                kernel_c00= 5,
                pool_c00= 4,
                c01= c2,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= c1,
                kernel_c10= 3,
                pool_c10= 2,
                c11= c2,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= c1,
                kernel_c20= 3,
                pool_c20= 2,
                c21= c2,
                kernel_c21= 3,
                pool_c21= 1 ,
                h1_out_factor=h1_out_factor,
                h2_out_factor=h2_out_factor,
                batch_size = batch_size,
                p=0.5)
    cost = model.cost(Y)

    save_path = '%sdata/%s/hier_derot/center/'%(dataset_path_prefix,setname)
    model_save_path = "%s%s.npy"%(save_path,model_save_path)
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
    print 'test cost', cost_nbatch/n_test_batches
    # numpy.save("%s%s_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h212_gm0_lm300_yt0_ep600.npy"%(save_path,dataset),uvd_norm)
    numpy.save("%s%s%s.npy"%(save_path,dataset,pred_save_name),uvd_norm)
if __name__ == '__main__':

    train_model(dataset='test',
                dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
                setname='icvl',
                source_name='_icvl_derot2_r0_r1_r2_uvd_bbox_21jnts_20151113_depth200',
                batch_size = 133,
                model_save_path='param_cost_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep2470',
                pred_save_name='_uvd_center_r0r1r2_c0016_c0132_c1016_c1132_c2016_c2132_h16_h216_gm0_lm300_yt0_ep2470')
    #
    # train_model(dataset='test',
    #             dataset_path_prefix='C:/Proj/Proj_CNN_Hier/',
    #             setname='nyu',
    #             source_name='_nyu_derot_shf_r0_r1_r2_uvd_bbox_21jnts_20151113_depth300',
    #             batch_size = 133,
    #             model_save_path='param_cost_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm300_yt0_ep1075',
    #             pred_save_name='_center_r0r1r2_uvd_c0016_c0132_c1016_c1132_c2016_c2132_h14_h216_gm0_lm300_yt0_ep1075')