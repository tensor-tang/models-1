import numpy as np
import tarfile
import gzip
import paddle.v2 as paddle
import math


eps = 1e-5
from_path = 'checkpoints/libri/params.latest.tar.gz'
to_path = 'checkpoints/libri/params.latest.bnFuse.tar.gz'


def updateWgt(num, conv_wgt, conv_bias, bn_var, bn_mean, bn_wgt, bn_bias):

    for oc in range(32):
        div_target = math.sqrt(bn_var[oc] + eps)
        a = bn_wgt[oc] / div_target
        b = bn_bias[oc] - bn_mean[oc] * bn_wgt[oc] / div_target
        # suppose weight has shape OIHW, num = I*H*W
        conv_bias[oc][0] = a * conv_bias[oc][0] + b
        for ks in range(num): 
            conv_wgt[oc*32 + ks] = a * conv_wgt[oc*32 + ks]


def updateFcWgt(d_dim, h_dim, fc_wgt, fc_bias, bn_var, bn_mean, bn_wgt, bn_bias):
    for i in range(h_dim):
        div_target = math.sqrt(bn_var[i] + eps)
        a = bn_wgt[i] /div_target
        b = bn_bias[i] - bn_mean[i] * bn_wgt[i] / div_target
        # suppose weight has shape OIHW, num = I*H*W
        fc_bias[i] = a * fc_bias[i] + b
        for j in range(d_dim):
            fc_wgt[j][i] = a * fc_wgt[j][i]

_parameters = paddle.parameters.Parameters.from_tar(gzip.open(from_path))

# print "%s" % (type(_parameters))
# print _parameters.names()
conv0_w = []
conv0_b = []
conv1_w = []
conv1_b = []
bn0_weight = []
bn0_mean = []
bn0_var = []
bn0_bias = []
bn1_weight = []
bn1_mean = []
bn1_var = []
bn1_bias = []
fc0_w = []
fc0_b = []
fc1_w = []
fc1_b = []
fc2_w = []
fc2_b = []
bn2_weight = []
bn2_mean = []
bn2_var = []
bn2_bias = []
bn3_weight = []
bn3_mean = []
bn3_var = []
bn3_bias = []
bn4_weight = []
bn4_mean = []
bn4_var = []
bn4_bias = []
fc0_w_dim = []
fc1_w_dim = []
fc2_w_dim = []

for param_key in _parameters.keys():
    param_shape = _parameters.get_shape(param_key)
    param = _parameters.__getitem__(param_key)

    # print param_key
    # print param_shape
    # print param

    if '___batch_norm_0__.w0' in param_key:
        bn0_weight = param
    elif '___batch_norm_0__.w1' in param_key:
        bn0_mean = param
    elif '___batch_norm_0__.w2' in param_key:
        bn0_var = param
    elif '___batch_norm_0__.wbias' in param_key:
        bn0_bias = param
    elif '___batch_norm_1__.w0' in param_key:
        bn1_weight = param
    elif '___batch_norm_1__.w1' in param_key:
        bn1_mean = param
    elif '___batch_norm_1__.w2' in param_key:
        bn1_var = param
    elif '___batch_norm_1__.wbias' in param_key:
        bn1_bias = param
    elif '___conv_0__.w0' in param_key:
        conv0_w = param
    elif '___conv_0__.wbias' in param_key:
        conv0_b = param
    elif '___conv_1__.w0' in param_key:
        conv1_w = param
    elif '___conv_1__.wbias' in param_key:
        conv1_b = param
    elif '___fc_layer_0__.w0' in param_key:
        fc0_w = param
        [dim_a, dim_b] = _parameters.get_shape(param_key)
        fc0_w_dim = [dim_a, dim_b] # dim::io
        print 'fc0 w', _parameters.get_shape(param_key)    
    elif '___fc_layer_0__.wbias' in param_key:
        fc0_b = param
        print 'fc0 b', _parameters.get_shape(param_key)      
    elif '___fc_layer_1__.w0' in param_key:
        fc1_w = param
        [dim_a, dim_b] = _parameters.get_shape(param_key)
        fc1_w_dim = [dim_a, dim_b]
        print 'fc1 w', _parameters.get_shape(param_key)    
    elif '___fc_layer_1__.wbias' in param_key:
        fc1_b = param
        print 'fc1 b', _parameters.get_shape(param_key)    
    elif '___fc_layer_2__.w0' in param_key:
        fc2_w = param
        [dim_a, dim_b] = _parameters.get_shape(param_key)
        fc2_w_dim = [dim_a, dim_b]
        print 'fc2 w', _parameters.get_shape(param_key)    
    elif '___fc_layer_2__.wbias' in param_key:
        fc2_b = param
        print 'fc2 b', _parameters.get_shape(param_key)
    elif '___batch_norm_2__.w0' in param_key:
        bn2_weight = param
    elif '___batch_norm_2__.w1' in param_key:
        bn2_mean = param
    elif '___batch_norm_2__.w2' in param_key:
        bn2_var = param
    elif '___batch_norm_2__.wbias' in param_key:
        bn2_bias = param
    elif '___batch_norm_3__.w0' in param_key:
        bn3_weight = param
    elif '___batch_norm_3__.w1' in param_key:
        bn3_mean = param
    elif '___batch_norm_3__.w2' in param_key:
        bn3_var = param
    elif '___batch_norm_3__.wbias' in param_key:
        bn3_bias = param
    elif '___batch_norm_4__.w0' in param_key:
        bn4_weight = param
    elif '___batch_norm_4__.w1' in param_key:
        bn4_mean = param
    elif '___batch_norm_4__.w2' in param_key:
        bn4_var = param
    elif '___batch_norm_4__.wbias' in param_key:
        bn4_bias = param


updateWgt(1*41*11, conv0_w[0], conv0_b, bn0_var[0], 
          bn0_mean[0], bn0_weight[0], bn0_bias[0])

updateWgt(32*21*11, conv1_w[0], conv1_b, bn1_var[0], 
          bn1_mean[0], bn1_weight[0], bn1_bias[0])

updateFcWgt(fc0_w_dim[0], fc0_w_dim[1], fc0_w, fc0_b[0], 
            bn2_var[0], bn2_mean[0], bn2_weight[0], bn2_bias[0])

updateFcWgt(fc1_w_dim[0], fc1_w_dim[1], fc1_w, fc1_b[0], 
            bn3_var[0], bn3_mean[0], bn3_weight[0], bn3_bias[0])

updateFcWgt(fc2_w_dim[0], fc2_w_dim[1], fc2_w, fc2_b[0], 
            bn4_var[0], bn4_mean[0], bn4_weight[0], bn4_bias[0])


for param_key in _parameters.keys():  
    if '___conv_0__.w0' in param_key:
        _parameters.__setitem__(param_key, conv0_w)
        print 'w0', _parameters.get_shape(param_key)
    elif '___conv_0__.wbias' in param_key:
        _parameters.__setitem__(param_key, conv0_b)
        print 'b0', _parameters.get_shape(param_key)
    elif '___conv_1__.w0' in param_key:
        _parameters.__setitem__(param_key, conv1_w)
        print 'w1', _parameters.get_shape(param_key)
    elif '___conv_1__.wbias' in param_key:
        _parameters.__setitem__(param_key, conv1_b)
        print 'b1', _parameters.get_shape(param_key)
        
    elif '___fc_layer_0__.w0' in param_key:
        _parameters.__setitem__(param_key, fc0_w)
    elif '___fc_layer_0__.wbias' in param_key:
        _parameters.__setitem__(param_key, fc0_b)      

    elif '___fc_layer_1__.w0' in param_key:
        _parameters.__setitem__(param_key, fc1_w)
    elif '___fc_layer_1__.wbias' in param_key:
        _parameters.__setitem__(param_key, fc1_b)              

    elif '___fc_layer_2__.w0' in param_key:
        _parameters.__setitem__(param_key, fc2_w)
    elif '___fc_layer_2__.wbias' in param_key:
        _parameters.__setitem__(param_key, fc2_b)              

print 'to save param'
with gzip.open(to_path, 'w') as f:
    _parameters.to_tar(f)


