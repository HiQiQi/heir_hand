__author__ = 'QiYE'

import numpy as np

from  Blocks import *


class CNN_Model(object):
    def __init__(self,
                 X,
                 model_info,
                 is_train,
        img_size = [96,96],
        c1 = 8,
        kernel_c1 = 5,
        pool_c1 = 4,
        c2 = 16,
        kernel_c2 = 3,
        pool_c2 = 2 ,
        batch_size = 64,
        p=0.5
        ):
        self.model_info = model_info
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',img_size
        layer0 = ConvPoolLayer(
            layer_name='conv_1',
            is_train=is_train,
            rng=rng,
            input=X,
            image_shape=(batch_size, 1, img_size[0], img_size[1]),
            filter_shape=(c1, 1, kernel_c1, kernel_c1),
            poolsize=(pool_c1, pool_c1),
            p=p
        )
        self.layers.append(layer0)
        self.params += layer0.params


        ft_shape = (img_size[0]-kernel_c1+1)/pool_c1
        print 'layer10 input', (ft_shape,ft_shape)
        layer01 = ConvPoolLayer(
            layer_name='conv_2',
            is_train=is_train,
            rng=rng,
            input=layer0.output,
            image_shape=(batch_size, c1,ft_shape, ft_shape),
            filter_shape=(c2, c1, kernel_c2, kernel_c2),
            poolsize=(pool_c2, pool_c2),
            p=p
        )
        self.layers.append(layer01)
        self.params += layer01.params


        ft_shape = (ft_shape  - kernel_c2+1)/pool_c2
        print 'layer10 input', (ft_shape,ft_shape)

        layer1_input = layer01.output.flatten(2)
        n_in = c2 * ft_shape *ft_shape
        layer1 = FullConLayer(
            layer_name='h1',
            is_train=is_train,
            rng=rng,
            input=layer1_input,
            n_in= n_in,
            n_out=n_in/2,
            activation='relu',
            p=p)
        self.layers.append(layer1)
        self.params += layer1.params

        layer2 = FullConLayer(
            layer_name='h2',
            is_train=is_train,
            rng=rng,
            input=layer1.output,
            n_in= n_in/2,
            n_out=n_in/4,
            activation='relu',
            p=p)
        self.layers.append(layer2)
        self.params += layer2.params


        layer3 = FullConLayer(
            layer_name='h_out',
            is_train=is_train,
            rng=rng,
            input=layer2.output,
            n_in= n_in/4,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.layers.append(layer3)
        self.params += layer3.params


    def cost(self,Y):
        return self.layers[-1].cost(Y)
    def save(self,path,c1,c2,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,gamma,lamda,yita,epoch),[params_value,train_cost,test_cost])
        print "%sparam_cost_%s_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,gamma,lamda,yita,epoch)

def func(x, a, x0, sigma):
    return a*T.exp(-(x-x0)**2/(2*sigma**2))
class CNN_Model_multi3_conv3(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,
                c02= 16,
                kernel_c02= 6,
                pool_c02= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,
                c12= 16,
                kernel_c12= 6,
                pool_c12= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,
                c22= 16,
                kernel_c22= 3,
                pool_c22= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        final_ft_shape_00 = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (final_ft_shape_00,final_ft_shape_00)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,final_ft_shape_00, final_ft_shape_00),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_01 = (final_ft_shape_00  - kernel_c01+1)/pool_c01
        print 'layer01 output', (final_ft_shape_01,final_ft_shape_01)

        layer02 = ConvPoolLayer(
            layer_name='conv_02',
            rng=rng,
            input=layer01.output,
            is_train=is_train,
            image_shape=(batch_size, c01,final_ft_shape_01, final_ft_shape_01),
            filter_shape=(c02, c01, kernel_c02, kernel_c02),
            poolsize=(pool_c02, pool_c02),
            p=p
        )
        self.hyper_params.append(['conv_02 filter_shape',(c02, c01, kernel_c02, kernel_c02)])
        self.hyper_params.append(['conv_02 poolsize',(pool_c02, pool_c02)])

        self.layers.append(layer02)
        self.params += layer02.params
        final_ft_shape_02 = (final_ft_shape_01  - kernel_c02+1)/pool_c02
        print 'layer02 output', (final_ft_shape_02,final_ft_shape_02)


        """""""""resolution 1"""""""""""""""
        print 'layer1 input ',(img_size_1,img_size_1)
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape_10 = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape_10,ft_shape_10)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape_10, ft_shape_10),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_11 = (ft_shape_10  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_11,final_ft_shape_11)

        layer12 = ConvPoolLayer(
            layer_name='conv_12',
            rng=rng,
            input=layer11.output,
            is_train=is_train,
            image_shape=(batch_size, c11,final_ft_shape_11, final_ft_shape_11),
            filter_shape=(c12, c11, kernel_c12, kernel_c12),
            poolsize=(pool_c12, pool_c12),
            p=p
        )
        self.hyper_params.append(['conv_12 filter_shape',(c12, c11, kernel_c12, kernel_c12)])
        self.hyper_params.append(['conv_12 poolsize',(pool_c12, pool_c12)])

        self.layers.append(layer12)
        self.params += layer12.params
        final_ft_shape_12 = (final_ft_shape_11  - kernel_c12+1)/pool_c12
        print 'layer12 output', (final_ft_shape_12,final_ft_shape_12)



        """""""""resolution 2"""""""""""""""
        print 'layer2 input ',(img_size_2,img_size_2)
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape_20 = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape_20,ft_shape_20)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape_20, ft_shape_20),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_21 = (ft_shape_20  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_21,final_ft_shape_21)

        layer22 = ConvPoolLayer(
            layer_name='conv_22',
            rng=rng,
            input=layer21.output,
            is_train=is_train,
            image_shape=(batch_size, c21,final_ft_shape_21, final_ft_shape_21),
            filter_shape=(c22, c21, kernel_c22, kernel_c22),
            poolsize=(pool_c22, pool_c22),
            p=p
        )
        self.hyper_params.append(['conv_22 filter_shape',(c22, c21, kernel_c22, kernel_c22)])
        self.hyper_params.append(['conv_22 poolsize',(pool_c22, pool_c22)])
        self.layers.append(layer22)
        self.params += layer22.params
        final_ft_shape_22 = (final_ft_shape_21  - kernel_c22+1)/pool_c22
        print 'conv_22 output', (final_ft_shape_22,final_ft_shape_22)

        layer1_input = T.concatenate([layer02.output.flatten(2),layer12.output.flatten(2),layer22.output.flatten(2)],axis=1)
        n_in = c02 * final_ft_shape_02 **2 + c12 *final_ft_shape_12**2 + c22*final_ft_shape_22**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/h2_out_factor,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model output', constants.OUT_DIM* constants.NUM_JNTS
        print 'model hyperparams ',self.hyper_params

    def sum_of_cost(self,Y,ratio,thresh):
        return self.cost(Y)+ratio*self.cost_bw_jnts(Y,thresh)

    def sum_of_cost_3(self,Y,ratio,thresh):
        return self.cost(Y)+ratio*self.cost_bw_jnts(Y,thresh)+ratio*self.cost_angle(Y)

    def cost(self,Y):
        return self.layers[-1].cost(Y)

    def cost_list(self,Y,ratio):
        diff = T.sqr(Y - self.layers[-1].output)
        cost_loc = T.sum(diff[:,0:3], axis=-1).mean()
        cost_rot = diff[:,3].mean()

        return [cost_loc+ratio*cost_rot,cost_loc,cost_rot]

    def cost_bw_jnts(self,Y,thresh):
        '''if the distance bw jnts smaller than the thresh --entry[2],give it a penalty of entry[2]-dist. the thresh is decided by the mean distance in training dataset'''
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        for entry in thresh:
            dist = T.sum(T.sqr(Y_[:,int(entry[0]),:]-Y_[:,int(entry[1]),:]),axis=-1)
            regu += T.mean(T.switch(entry[2]-dist>0,entry[2]-dist,0))
        return regu/len(thresh)

    def cost_angle(self,Y):
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        v0 = Y_[:,0,:] - Y_[:,3,:]
        nv0 =T.sum(v0*v0,axis=-1)
        v1 = Y_[:,2,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.55**2,0,inner))


        v1 = Y_[:,4,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.55**2,0,inner))

        v1 = Y_[:,5,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.68**2,0,inner))
        return regu/3


    def regularization(self,Y,gaussian_param_btw_jnts):
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        for entry in gaussian_param_btw_jnts:
            dist = T.sum(T.sqr(Y_[:,int(entry[0]),:]-Y_[:,int(entry[1]),:]),axis=-1)
            regu +=T.mean(( entry[2] - func(dist,a=entry[2],x0=entry[3],sigma=entry[4]))/entry[2])

        return regu/len(gaussian_param_btw_jnts)


    def save(self,path,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)

class CNN_Model_multi3(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        print 'layer1 input ',(img_size_1,img_size_1)
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        print 'layer2 input ',(img_size_2,img_size_2)
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)



        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_0 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/h2_out_factor,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model output', constants.OUT_DIM* constants.NUM_JNTS
        print 'model hyperparams ',self.hyper_params

    def sum_of_cost(self,Y,ratio,thresh):
        return self.cost(Y)+ratio*self.cost_bw_jnts(Y,thresh)

    def sum_of_cost_3(self,Y,ratio,thresh):
        return self.cost(Y)+ratio*self.cost_bw_jnts(Y,thresh)+ratio*self.cost_angle(Y)

    def cost(self,Y):
        return self.layers[-1].cost(Y)

    def cost_list(self,Y,ratio):
        diff = T.sqr(Y - self.layers[-1].output)
        cost_loc = T.sum(diff[:,0:3], axis=-1).mean()
        cost_rot = diff[:,3].mean()

        return [cost_loc+ratio*cost_rot,cost_loc,cost_rot]

    def cost_bw_jnts(self,Y,thresh):
        '''if the distance bw jnts smaller than the thresh --entry[2],give it a penalty of entry[2]-dist. the thresh is decided by the mean distance in training dataset'''
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        for entry in thresh:
            dist = T.sum(T.sqr(Y_[:,int(entry[0]),:]-Y_[:,int(entry[1]),:]),axis=-1)
            regu += T.mean(T.switch(entry[2]-dist>0,entry[2]-dist,0))
        return regu/len(thresh)

    def cost_angle(self,Y):
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        v0 = Y_[:,0,:] - Y_[:,3,:]
        nv0 =T.sum(v0*v0,axis=-1)
        v1 = Y_[:,2,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.55**2,0,inner))


        v1 = Y_[:,4,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.55**2,0,inner))

        v1 = Y_[:,5,:] - Y_[:,3,:]
        nv1 = T.sum(v1*v1,axis=-1)
        inner = T.sqr(T.sum(v1*v0,axis=-1))/nv0/nv1
        regu += T.mean(T.switch(inner <0.68**2,0,inner))
        return regu/3


    def regularization(self,Y,gaussian_param_btw_jnts):
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        for entry in gaussian_param_btw_jnts:
            dist = T.sum(T.sqr(Y_[:,int(entry[0]),:]-Y_[:,int(entry[1]),:]),axis=-1)
            regu +=T.mean(( entry[2] - func(dist,a=entry[2],x0=entry[3],sigma=entry[4]))/entry[2])

        return regu/len(gaussian_param_btw_jnts)


    def save(self,path,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)


class CNN_Model_multi3_conv1(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                 num_channel=16,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,


                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,


                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, num_channel, img_size_0, img_size_0),
            filter_shape=(c00, num_channel, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        final_ft_shape_00 = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (final_ft_shape_00,final_ft_shape_00)



        """""""""resolution 1"""""""""""""""
        print 'layer1 input ',(img_size_1,img_size_1)
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, num_channel, img_size_1, img_size_1),
            filter_shape=(c10, num_channel, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        final_ft_shape_10 = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (final_ft_shape_10,final_ft_shape_10)


        """""""""resolution 2"""""""""""""""
        print 'layer2 input ',(img_size_2,img_size_2)
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, num_channel, img_size_2, img_size_2),
            filter_shape=(c20, num_channel, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        final_ft_shape_20 = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (final_ft_shape_20,final_ft_shape_20)



        layer1_input = T.concatenate([layer00.output.flatten(2),layer10.output.flatten(2),layer20.output.flatten(2)],axis=1)
        n_in = c00 * final_ft_shape_00 **2 + c10 *final_ft_shape_10**2 + c20*final_ft_shape_20**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/h2_out_factor,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model output', constants.OUT_DIM* constants.NUM_JNTS
        print 'model hyperparams ',self.hyper_params


    def cost(self,Y):
        return self.layers[-1].cost(Y)

    def save(self,path,c00,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)


class CNN_Model_multi4(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 X3,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                img_size_3=44,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                c30= 8,
                kernel_c30= 3,
                pool_c30= 2,
                c31= 16,
                kernel_c31= 3,
                pool_c31= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        print 'layer1 input ',(img_size_1,img_size_1)
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        print 'layer2 input ',(img_size_2,img_size_2)
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)

        """""""""resolution whole"""""""""""""""
        print 'layer3 input ',(img_size_3,img_size_3)
        layer30 = ConvPoolLayer(
            layer_name='conv_30',
            rng=rng,
            input=X3,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_3, img_size_3),
            filter_shape=(c30, 1, kernel_c30, kernel_c30),
            poolsize=(pool_c30, pool_c30),
            p=p
        )
        self.hyper_params.append(['conv_30 filter_shape',(c30, 1, kernel_c30, kernel_c30)])
        self.hyper_params.append(['conv_30 poolsize',(pool_c30, pool_c30)])
        self.layers.append(layer30)
        self.params += layer30.params

        ft_shape = (img_size_3-kernel_c30+1)/pool_c30
        print 'layer30 output', (ft_shape,ft_shape)


        layer31 = ConvPoolLayer(
            layer_name='conv_31',
            rng=rng,
            input=layer30.output,
            is_train=is_train,
            image_shape=(batch_size, c30,ft_shape, ft_shape),
            filter_shape=(c31, c30, kernel_c21, kernel_c31),
            poolsize=(pool_c31, pool_c31),
            p=p
        )
        self.hyper_params.append(['conv_31 filter_shape',(c31, c30, kernel_c31, kernel_c31)])
        self.hyper_params.append(['conv_31 poolsize',(pool_c31, pool_c31)])
        self.layers.append(layer31)
        self.params += layer31.params
        final_ft_shape_3 = (ft_shape  - kernel_c31+1)/pool_c31
        print 'layer31 output', (final_ft_shape_3,final_ft_shape_3)


        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2),layer31.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_0 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2+c31*final_ft_shape_3**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/h2_out_factor,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model output', constants.OUT_DIM* constants.NUM_JNTS

    def sum_of_cost(self,Y,ratio,gaussian_param_btw_jnts):
        return (1-ratio)*self.cost(Y)+ratio*self.regularization(Y,gaussian_param_btw_jnts)

    def cost(self,Y):
        return self.layers[-1].cost(Y)
    def cost_list(self,Y,ratio):
        diff = T.sqr(Y - self.layers[-1].output)
        cost_loc = T.sum(diff[:,0:3], axis=-1).mean()
        cost_rot = diff[:,3].mean()

        return [cost_loc+ratio*cost_rot,cost_loc,cost_rot]


    def regularization(self,Y,gaussian_param_btw_jnts):
        Y_ = T.reshape(Y,(Y.shape[0], constants.NUM_JNTS, constants.OUT_DIM ))
        regu=0
        for entry in gaussian_param_btw_jnts:
            dist = T.sum(T.sqr(Y_[:,int(entry[0]),:]-Y_[:,int(entry[1]),:]),axis=-1)
            regu +=T.mean(( entry[2] - func(dist,a=entry[2],x0=entry[3],sigma=entry[4]))/entry[2])

        return regu/len(gaussian_param_btw_jnts)


    def save(self,path,c00,c01,c10,c11,c20,c21,c30,c31,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_c30%d_c31%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,c30,c31,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_c30%d_c31%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,c30,c31,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)
class CNN_Model_multi3_regress_class_concate(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)
        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_2 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = Comp_Class_Regress(
            input=layer2.output,
            n_in= n_in/h2_out_factor,
            n_outs=[constants.OUT_DIM* constants.NUM_JNTS, constants.Num_Class])
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model regressor output', constants.OUT_DIM* constants.NUM_JNTS, 'classifier', constants.Num_Class


    def cost_list(self,Y):
        return self.layers[-1].cost(Y)

    def save(self,path,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c1%d_c2%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c1%d_c2%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)

class CNN_Model_multi3_regress_class(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)



        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_2 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor

        layer3 = CompLayer(
            layer_name='comp_out',
            num_layers=2,
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/h2_out_factor,
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.OUT_DIM* constants.NUM_JNTS)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'model output', constants.OUT_DIM* constants.NUM_JNTS

    def sum_of_cost(self,Y,Rot):
        return reduce(lambda x, y: x+y,self.cost_list(Y,Rot))

    def cost_list(self,Y,Rot):
        return self.layers[-1].cost(Y,Rot)

    def save(self,path,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c1%d_c2%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c1%d_c2%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)
class CNN_Model_multi2_htmap(object):
    def __init__(self,
                 X0,
                 X1,
                is_train,
                img_size_0 = 96,
                img_size_1=48,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                batch_size = 64,
                p=0.5

        ):
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=0.5
        )
        self.hyper_params.append(['conv_1 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_1 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=0.5
        )
        self.hyper_params.append(['conv_00 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=0.5
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])

        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=0.5
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])
        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)


        layer1_input = T.concatenate([layer01.output,layer11.output,],axis=1).flatten(2)
        n_in = c01 * final_ft_shape_0 **2 + c11 *final_ft_shape_1**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            is_train=is_train,
            input=layer1_input,
            n_in= n_in,
            n_out=n_in/2,
            activation='relu',
            p=p)
        self.layers.append(layer1)
        self.params += layer1.params
        self.hyper_params.append(['h1 n_in n_out',(n_in, n_in/2)])
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            is_train=is_train,
            input=layer1.output,
            n_in= n_in/2,
            n_out=n_in/4,
            activation='relu',
            p=p)
        self.layers.append(layer2)
        self.params += layer2.params
        self.hyper_params.append(['h2 n_in n_out',(n_in/2, n_in/4)])

        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/4,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.layers.append(layer3)
        self.params += layer3.params
        self.hyper_params.append(['h_out n_in n_out',(n_in/4, constants.OUT_DIM* constants.NUM_JNTS)])

    def cost(self,Y):
        return self.layers[-1].cost(Y)

    def save(self,path,c1,c2,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sjnt_htmap_param_cost_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,c1,c2,gamma,lamda,yita,epoch),[params_value,train_cost,test_cost])

class CNN_Model_multi2(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                is_train,
                img_size_0 = 96,
                img_size_1=48,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                batch_size = 64,
                p=0.5,


        ):
        self.model_info=model_info
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_1',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=0.5
        )
        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_2',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=0.5
        )
        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_1',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=0.5
        )
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_2',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=0.5
        )
        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)


        layer1_input = T.concatenate([layer01.output,layer11.output,],axis=1).flatten(2)
        n_in = c01 * final_ft_shape_0 **2 + c11 *final_ft_shape_1**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            is_train=is_train,
            input=layer1_input,
            n_in= n_in,
            n_out=n_in/8,
            activation='relu',
            p=p)
        self.layers.append(layer1)
        self.params += layer1.params

        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            is_train=is_train,
            input=layer1.output,
            n_in= n_in/8,
            n_out=n_in/16,
            activation='relu',
            p=p)
        self.layers.append(layer2)
        self.params += layer2.params


        layer3 = FullConLayer(
            layer_name='h_out',
            rng=rng,
            input=layer2.output,
            is_train=is_train,
            n_in= n_in/16,
            n_out=constants.OUT_DIM* constants.NUM_JNTS,
            activation=None,
            p=p)
        self.layers.append(layer3)
        self.params += layer3.params


    def cost(self,Y):
        return self.layers[-1].cost(Y)

    def save(self,path,c1,c2,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,gamma,lamda,yita,epoch),[params_value,train_cost,test_cost])
        print "%sparam_cost_%s_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c1,c2,gamma,lamda,yita,epoch)

class CNN_Model_multi2_softmax(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,

                 is_train,
                img_size_0 = 96,
                img_size_1=48,

                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)




        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_0 **2 + c11 *final_ft_shape_1**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor


        layer3 = LogisticRegression(
            layer_name='h_out',
            input=layer2.output,
            n_in= n_in/h2_out_factor,
            n_out=constants.Num_Class
        )
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.Num_Class)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'h2 output', constants.Num_Class


    def cost(self,Y):
        return self.layers[-1].negative_log_likelihood(Y)

    def save(self,path,c00,c01,c10,c11,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)


class CNN_Model_multi3_softmax(object):
    def __init__(self,
                 model_info,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                h1_out_factor=9,
                h2_out_factor=18,

                batch_size = 64,
                p=0.5

        ):
        self.model_info= model_info
        self.hyper_params=[]
        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_00',
            rng=rng,
            input=X0,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.hyper_params.append(['conv_00 filter_shape',(c00, 1, kernel_c00, kernel_c00)])
        self.hyper_params.append(['conv_00 poolsize',(pool_c00, pool_c00)])

        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_01',
            rng=rng,
            input=layer00.output,
            is_train=is_train,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.hyper_params.append(['conv_01 filter_shape',(c01, c00, kernel_c01, kernel_c01)])
        self.hyper_params.append(['conv_01 poolsize',(pool_c01, pool_c01)])

        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_10',
            rng=rng,
            input=X1,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.hyper_params.append(['conv_10 filter_shape',(c10, 1, kernel_c10, kernel_c10)])
        self.hyper_params.append(['conv_10 poolsize',(pool_c10, pool_c10)])
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_11',
            rng=rng,
            input=layer10.output,
            is_train=is_train,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.hyper_params.append(['conv_11 filter_shape',(c11, c10, kernel_c11, kernel_c11)])
        self.hyper_params.append(['conv_11 poolsize',(pool_c11, pool_c11)])

        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        layer20 = ConvPoolLayer(
            layer_name='conv_20',
            rng=rng,
            input=X2,
            is_train=is_train,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.hyper_params.append(['conv_20 filter_shape',(c20, 1, kernel_c20, kernel_c20)])
        self.hyper_params.append(['conv_20 poolsize',(pool_c20, pool_c20)])
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_21',
            rng=rng,
            input=layer20.output,
            is_train=is_train,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.hyper_params.append(['conv_21 filter_shape',(c21, c20, kernel_c21, kernel_c21)])
        self.hyper_params.append(['conv_21 poolsize',(pool_c21, pool_c21)])
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)



        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_2 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            input=layer1_input,
            is_train=is_train,
            n_in= n_in,
            n_out=n_in/h1_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h1 n_in n_out',(n_in,n_in/h1_out_factor)])
        self.layers.append(layer1)
        self.params += layer1.params
        print 'h1 output', n_in/h1_out_factor
        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            input=layer1.output,
            is_train=is_train,
            n_in= n_in/h1_out_factor,
            n_out=n_in/h2_out_factor,
            activation='relu',
            p=p)
        self.hyper_params.append(['h2 n_in n_out',(n_in/h1_out_factor,n_in/h2_out_factor)])
        self.layers.append(layer2)
        self.params += layer2.params
        print 'h2 output', n_in/h2_out_factor


        layer3 = LogisticRegression(
            layer_name='h_out',
            input=layer2.output,
            n_in= n_in/h2_out_factor,
            n_out=constants.Num_Class
        )
        self.hyper_params.append(['h2 n_in n_out',(n_in/h2_out_factor, constants.Num_Class)])
        self.layers.append(layer3)
        self.params += layer3.params
        print 'h2 output', constants.Num_Class


    def cost(self,Y):
        return self.layers[-1].negative_log_likelihood(Y)

    def save(self,path,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch),[params_value,self.hyper_params,train_cost,test_cost])
        print "%sparam_cost_%s_c00%d_c01%d_c10%d_c11%d_c20%d_c21%d_h1%d_h2%d_gm%d_lm%d_yt%d_ep%d"%(path,self.model_info,c00,c01,c10,c11,c20,c21,h1_out_factor,h2_out_factor,gamma,lamda,yita,epoch)

class CNN_Model_multi3_softmax_group(object):
    def __init__(self,
                 X0,
                 X1,
                 X2,
                 is_train,
                img_size_0 = 96,
                img_size_1=48,
                img_size_2=24,
                c00= 8,
                kernel_c00= 5,
                pool_c00= 4,
                c01= 16,
                kernel_c01= 6,
                pool_c01= 2 ,

                c10= 8,
                kernel_c10= 5,
                pool_c10= 2,
                c11= 16,
                kernel_c11= 6,
                pool_c11= 2 ,

                c20= 8,
                kernel_c20= 3,
                pool_c20= 2,
                c21= 16,
                kernel_c21= 3,
                pool_c21= 1 ,

                batch_size = 64,
                p=0.5

        ):

        self.layers = []
        self.params = []
        rng = np.random.RandomState(2391)

        # y.tag.test_value = train_set_y.get_value()
        # theano.printing.Print('x0')(x0.shape)
        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the symbolic model'
        print 'layer0 input ',(img_size_0,img_size_0)
        layer00 = ConvPoolLayer(
            layer_name='conv_1',
            rng=rng,
            is_train=is_train,
            input=X0,
            image_shape=(batch_size, 1, img_size_0, img_size_0),
            filter_shape=(c00, 1, kernel_c00, kernel_c00),
            poolsize=(pool_c00, pool_c00),
            p=p
        )
        self.layers.append(layer00)
        self.params += layer00.params

        ft_shape = (img_size_0-kernel_c00+1)/pool_c00
        print 'layer00 output', (ft_shape,ft_shape)

        layer01 = ConvPoolLayer(
            layer_name='conv_2',
            rng=rng,
            is_train=is_train,
            input=layer00.output,
            image_shape=(batch_size, c00,ft_shape, ft_shape),
            filter_shape=(c01, c00, kernel_c01, kernel_c01),
            poolsize=(pool_c01, pool_c01),
            p=p
        )
        self.layers.append(layer01)
        self.params += layer01.params
        final_ft_shape_0 = (ft_shape  - kernel_c01+1)/pool_c01
        print 'layer10 output', (final_ft_shape_0,final_ft_shape_0)


        """""""""resolution 1"""""""""""""""
        layer10 = ConvPoolLayer(
            layer_name='conv_1',
            rng=rng,
            is_train=is_train,
            input=X1,
            image_shape=(batch_size, 1, img_size_1, img_size_1),
            filter_shape=(c10, 1, kernel_c10, kernel_c10),
            poolsize=(pool_c10, pool_c10),
            p=p
        )
        self.layers.append(layer10)
        self.params += layer10.params

        ft_shape = (img_size_1-kernel_c10+1)/pool_c10
        print 'layer10 output', (ft_shape,ft_shape)

        layer11 = ConvPoolLayer(
            layer_name='conv_2',
            rng=rng,
            is_train=is_train,
            input=layer10.output,
            image_shape=(batch_size, c10,ft_shape, ft_shape),
            filter_shape=(c11, c10, kernel_c11, kernel_c11),
            poolsize=(pool_c11, pool_c11),
            p=p
        )
        self.layers.append(layer11)
        self.params += layer11.params
        final_ft_shape_1 = (ft_shape  - kernel_c11+1)/pool_c11
        print 'layer11 output', (final_ft_shape_1,final_ft_shape_1)

        """""""""resolution 2"""""""""""""""
        layer20 = ConvPoolLayer(
            layer_name='conv_1',
            rng=rng,
            is_train=is_train,
            input=X2,
            image_shape=(batch_size, 1, img_size_2, img_size_2),
            filter_shape=(c20, 1, kernel_c20, kernel_c20),
            poolsize=(pool_c20, pool_c20),
            p=p
        )
        self.layers.append(layer20)
        self.params += layer20.params

        ft_shape = (img_size_2-kernel_c20+1)/pool_c20
        print 'layer20 output', (ft_shape,ft_shape)


        layer21 = ConvPoolLayer(
            layer_name='conv_2',
            rng=rng,
            is_train=is_train,
            input=layer20.output,
            image_shape=(batch_size, c20,ft_shape, ft_shape),
            filter_shape=(c21, c20, kernel_c21, kernel_c21),
            poolsize=(pool_c21, pool_c21),
            p=p
        )
        self.layers.append(layer21)
        self.params += layer21.params
        final_ft_shape_2 = (ft_shape  - kernel_c21+1)/pool_c21
        print 'layer21 output', (final_ft_shape_2,final_ft_shape_2)



        layer1_input = T.concatenate([layer01.output.flatten(2),layer11.output.flatten(2),layer21.output.flatten(2)],axis=1)
        n_in = c01 * final_ft_shape_2 **2 + c11 *final_ft_shape_1**2 + c21*final_ft_shape_2**2

        print 'full connected input', n_in
        layer1 = FullConLayer(
            layer_name='h1',
            rng=rng,
            is_train=is_train,
            input=layer1_input,
            n_in= n_in,
            n_out=n_in/2,
            activation='relu',
            p=p)
        self.layers.append(layer1)
        self.params += layer1.params

        layer2 = FullConLayer(
            layer_name='h2',
            rng=rng,
            is_train=is_train,
            input=layer1.output,
            n_in= n_in/2,
            n_out=n_in/4,
            activation='relu',
            p=p)
        self.layers.append(layer2)
        self.params += layer2.params


        layer3 = LogisticRegression(
            layer_name='h_out',
            input=layer2.output,
            n_in= n_in/4,
            n_out=constants.OUT_DIM* constants.NUM_JNTS
        )
        self.layers.append(layer3)
        self.params += layer3.params


    def cost(self,Y):
        return self.layers[-1].negative_log_likelihood(Y)

    def save(self,path,c1,c2,gamma,lamda,yita,epoch,train_cost,test_cost):
        params_value = []
        for param_i in self.params:
            params_value.append(param_i.get_value())
        np.save("%sjnt_discrete_center_param_cost_c1%d_c2%d_gm%d_lm%d_yt%d_ep%d"%(path,c1,c2,gamma,lamda,yita,epoch),[params_value,train_cost,test_cost])