# -*- coding: utf-8 -*-
"""

This code is to generate the json files, they make optimization easier and quicker
It also explains how the paramters should be used

@author: laramos
"""
import json

params = dict()

params['conv_type'] = 'LSTM'
params['num_conv_layers'] = 3
params['num_filters'] = [32,32,32] #number of feature maps per layer
params['kernel_size'] = [[1,32],[1,32],[1,32]]
params['activation'] = 'relu'#activation for all other layers but the last
params['activation_last'] = 'sigmoid' #Activation for the last layer
params['dropout'] = [0.3,0.3,0.3,0.6] #set to 0 for each conv/dense layer if not used, for instance 3 conv layers without droppout in between = [0,0,0], this will not add dropout between
params['num_dense_layers'] = 1 #total number of dense layer
params['size_dense_layer'] = [256] #number of neurons per layer, should have one value per layer otherwise you'll get an error
params['num_classes'] = 2 #Classes in your dataset
params['pool_size'] = [[0],[0],[0]]

import json

with open(params['conv_type']+'_model1.json', 'w') as f:
    json_string = json.dump(params,f)



params = dict()
params['conv_type'] = 'LSTM'
params['num_conv_layers'] = 4
params['num_filters'] = [16,16,32,64] 
params['kernel_size'] = [[1,32],[1,32],[1,16],[1,16]]
params['activation'] = 'relu'
params['activation_last'] = 'sigmoid' 
params['dropout'] = [0.3,0.3,0.3,0.3,0.6] 
params['num_dense_layers'] = 1
params['size_dense_layer'] = [256] 
params['num_classes'] = 2 
params['pool_size'] = [[1,1,2],[0],[1,1,2],[0]]
import json

with open(params['conv_type']+'_model2.json', 'w') as f:
    json_string = json.dump(params,f)
    
    
params = dict()
params['conv_type'] = 'LSTM'
params['num_conv_layers'] = 3
params['num_filters'] = [32,32,32] 
params['kernel_size'] = [[3,3],[3,3],[3,3]]
params['activation'] = 'relu'
params['activation_last'] = 'sigmoid' 
params['dropout'] = [0.3,0.3,0.3,0.6] 
params['num_dense_layers'] = 1
params['size_dense_layer'] = [256] 
params['num_classes'] = 2 
params['pool_size'] = [[1,1,2],[0],[1,1,2]]

import json

with open(params['conv_type']+'_model3.json', 'w') as f:
    json_string = json.dump(params,f)
    
    
params = dict()
params['conv_type'] = 'CNN'
params['num_conv_layers'] = 3
params['num_filters'] = [32,32,32] 
params['kernel_size'] = [[1,36],[1,36],[1,36]]
params['activation'] = 'relu'
params['activation_last'] = 'sigmoid' 
params['dropout'] = [0.3,0.3,0.3,0.6] 
params['num_dense_layers'] = 1
params['size_dense_layer'] = [256]
params['num_classes'] = 2 
params['pool_size'] = [[0],[0],[0]]

import json

with open(params['conv_type']+'_model1.json', 'w') as f:
    json_string = json.dump(params,f) 
    
    
params = dict()
params['conv_type'] = 'CNN'
params['num_conv_layers'] = 3
params['num_filters'] = [16,32,64] 
params['kernel_size'] = [[1,32],[1,32],[1,32]]
params['activation'] = 'relu'
params['activation_last'] = 'sigmoid' 
params['dropout'] = [0.2,0.2,0.2,0.5] 
params['num_dense_layers'] = 2
params['size_dense_layer'] = [128,256]
params['num_classes'] = 2 
params['pool_size'] = [[1,2],[1,2],[1,2]]

import json

with open(params['conv_type']+'_model2.json', 'w') as f:
    json_string = json.dump(params,f)        
    
    
    
params = dict()
params['conv_type'] = 'CNN'
params['num_conv_layers'] = 4
params['num_filters'] = [16,16,32,64] 
params['kernel_size'] = [[2,4],[2,4],[2,4],[2,4]]
params['activation'] = 'relu'
params['activation_last'] = 'sigmoid' 
params['dropout'] = [0.2,0.2,0.2,0.5] 
params['num_dense_layers'] = 2
params['size_dense_layer'] = [128,256]
params['num_classes'] = 2 
params['pool_size'] = [[1,2],[1,2],[1,2],[1,2]]

import json

with open(params['conv_type']+'_model3.json', 'w') as f:
    json_string = json.dump(params,f)     
    
    
