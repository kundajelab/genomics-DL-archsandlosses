import genomicsdlarchsandlosses.bpnet.bpnetdefaults as bpnetdefaults
import numpy as np

from decimal import Decimal
from genomicsdlarchsandlosses.bpnet.archs import BPNet
from tensorflow.keras.utils import plot_model

motif_module_params = bpnetdefaults.MOTIF_MODULE_PARAMS
syntax_module_params = bpnetdefaults.SYNTAX_MODULE_PARAMS
profile_head_params = bpnetdefaults.PROFILE_HEAD_PARAMS
counts_head_params = bpnetdefaults.COUNTS_HEAD_PARAMS
profile_bias_module_params = bpnetdefaults.PROFILE_BIAS_MODULE_PARAMS
counts_bias_module_params = bpnetdefaults.COUNTS_BIAS_MODULE_PARAMS

# change activations to linear
# motif_module_params['activation'] = 'linear'
# syntax_module_params['activation'] = 'linear'
# profile_head_params['activation'] = 'linear'

# tasks = {
#     0:{
#         'signal': {
#             'source': ['unstranded.bw']
#         },
#         'loci': {
#             'source': ['peaks.bed']
#         },
#         'bias': {
#             'source': ['bias.bw'],
#             'smoothing': [[7.0, 81]]
#         }        
#     }
# } 

tasks = {
    0: {
        'signal': {
            'source': ['plus.bw', 'minus.bw']
        },
        'loci': {
            'source': ['peaks.bed']
        },
        'bias': {
            'source': ['genome_wide_bias_plus.bw', 'genome_wide_bias_minus.bw'],
            'smoothing': [None, None]
        }
    },
    1: {
        'signal': {
            'source': ['unstranded.bw']
        },
        'loci': {
            'source': ['peaks.bed']
        },
        'bias': {
            'source': [],
            'smoothing': []
        }
    },
    2: {
        'signal': {
            'source': ['unstranded.bw']
        },
        'loci': {
            'source': ['peaks.bed']
        },
        'bias': {
            'source': ['unstranded_genome_wide_bias.bw'],
            'smoothing': [[7.0, 81]]
        }
    },
    3: {
        'signal': {
            'source': ['plus.bw', 'minus.bw']
        },
        'loci': {
            'source': ['peaks.bed']
        },
        'bias': {
            'source': ['genome_wide_bias_plus.bw', 'genome_wide_bias_minus.bw'],
            'smoothing': [None, [7.0, 81]]
        }
    },
    4: {
        'signal': {
            'source': ['plus.bw', 'minus.bw']
        },
        'loci': {
            'source': ['peaks.bed']
        },
        'bias': {
            'source': ['genome_wide_bias_plus.bw', 'genome_wide_bias_minus.bw'],
            'smoothing': [[7.0, 81], [7.0, 81]]
        } 
    }
}

syntax_module_params['pre_activation_residual_unit'] = False
profile_bias_module_params['kernel_sizes'] = [1, 1, 1, 1, 1]


# single unstranded task with linear activations
model_v1 = BPNet(
    tasks,
    syntax_module_params=syntax_module_params,
    profile_bias_module_params=profile_bias_module_params)

# change padding to 'same'
motif_module_params['padding'] = 'same'
syntax_module_params['padding'] = 'same'
profile_head_params['padding'] = 'same'
model_v2 = BPNet(
    tasks,
    motif_module_params=motif_module_params, 
    syntax_module_params=syntax_module_params, 
    profile_head_params=profile_head_params, 
    profile_bias_module_params=profile_bias_module_params)

model_v1.summary()
model_v2.summary()
plot_model(
    model_v1, to_file='model_v1.png', show_shapes=True, show_layer_names=True)
plot_model(
    model_v2, to_file='model_v2.png', show_shapes=True, show_layer_names=True)

def set_all_weights_to_one(model):
    """
        Set weights in all layers of the model to 1
        
        Args:
            model (tensorflow.keras.Model): model object
            
    """
    
    # set all weights (W) and biases (B) <<- (Wx + B) to 1
    for layer in model.layers:
        if layer.count_params() == 0:
            continue
        weights_n_biases = layer.get_weights()
        weights = weights_n_biases[0]
        biases = weights_n_biases[1]
        _weights = np.ones(weights.shape)
        _biases = np.ones(biases.shape)
        layer.set_weights((_weights, _biases))
    
def print_all_weights(model):
    """
        Print weights of all layers of the model
        
        Args:
            model (tensorflow.keras.Model): model object
            
    """    
    
    for layer in model.layers:
        if layer.count_params() == 0:
            continue
        weights_n_biases = layer.get_weights()
        weights = weights_n_biases[0]
        biases = weights_n_biases[1]
        print("{} - Weights \n {}".format(layer.name, weights))
        print("{} - Biases \n {}".format(layer.name, biases))
        
# #print_all_weights(model_v1)
set_all_weights_to_one(model_v1)
# #print_all_weights(model_v1)

# #print_all_weights(model_v2)
set_all_weights_to_one(model_v2)
# #print_all_weights(model_v2)

one_hot_input = np.ones((1, 2114, 4))
profile_bias_input_0 = np.ones((1, 1000, 2))
profile_bias_input_2 = np.ones((1, 1000, 2))
profile_bias_input_3 = np.ones((1, 1000, 3))
profile_bias_input_4 = np.ones((1, 1000, 4))
counts_bias_input_0 = np.ones((1, 2))
counts_bias_input_2 = np.ones((1, 2))
counts_bias_input_3 = np.ones((1, 3))
counts_bias_input_4 = np.ones((1, 4))
model_input = {
    'sequence': one_hot_input,
    'profile_bias_input_0': profile_bias_input_0,
    'profile_bias_input_2': profile_bias_input_2,
    'profile_bias_input_3': profile_bias_input_3,
    'profile_bias_input_4': profile_bias_input_4,
    'counts_bias_input_0': counts_bias_input_0,
    'counts_bias_input_2': counts_bias_input_2,
    'counts_bias_input_3': counts_bias_input_3,
    'counts_bias_input_4': counts_bias_input_4
}
predictions_v1 = model_v1.predict(model_input)
predictions_v2 = model_v2.predict(model_input)

print("Model 1 (profile sum)", Decimal(np.sum(predictions_v1[0]).astype(float)))
print("Model 2 (profile sum)", Decimal(np.sum(predictions_v2[0]).astype(float)))
print("Model 1 (counts ouput)",Decimal(np.sum(predictions_v1[1]).astype(float)))
print("Model 2 (counts ouput)", Decimal(np.sum(predictions_v2[1]).astype(float)))
