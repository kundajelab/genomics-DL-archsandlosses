"""
    This module contains all the fucntions that define various
    bpnet network architectures

    Fucntions:
    
        BPNet: The network architecture for BPNet as described in 
            the paper: 
            https://www.biorxiv.org/content/10.1101/737981v1.full.pdf
"""

from genomicsDLarchsandlosses.bpnet.attribution_prior \
    import AttributionPriorModel
from genomicsDLarchsandlosses.utils.exceptionhandler \
    import NoTracebackException
from tensorflow.keras import layers, Model
from tensorflow.keras.backend import int_shape

def BPNet(
    input_seq_len=2114, output_len=1000, num_bias_profiles=2, filters=64,
    num_dilation_layers=8, conv1_kernel_size=21, dilation_kernel_size=3, 
    profile_kernel_size=75, num_tasks=2, use_attribution_prior=False,
    attribution_prior_params=None):
    
    """
        BPNet model architecture as described in the BPNet paper
        https://www.biorxiv.org/content/10.1101/737981v1.full.pdf
        
        Args:
            input_seq_len (int): length of input DNA sequence
            
            output_len (int): length of the profile output
            
            num_bias_profiles (int): total number of control/bias
                tracks. In the case where original control and one  
                smoothed version are provided this value is 2.
            
            filters (int): number of filters in each convolutional
                layer of BPNet
                
            num_dilation_layers (int): num of layers with dilated
                convolutions
            
            conv1_kernel_size (int): kernel size for the first 1D 
                convolution
            
            dilation_kernel_size (int): kernel size in each of the
                dilation layers
                
            profile_kernel_size (int): kernel size in the first 
                convolution of the profile head branch of the network
            
            num_tasks (int): number of output profile tracks
            
            use_attribution_prior (bool): indicate whether to use 
                attribution prior model
                
            attribution_prior_params (dict): python dictionary with 
                keys 'frequency_limit', 'limit_softness' & 
                'smooth_sigma'
            
        Returns:
            keras.model.Model
        
    """

    # The three inputs to BPNet
    inp = layers.Input(shape=(input_seq_len, 4), name='sequence')
    
    bias_counts_input = layers.Input(shape=(1, ), name="control_logcount")
    
    bias_profile_input = layers.Input(
        shape=(output_len, num_bias_profiles), name="control_profile")
    # end inputs

    # first convolution without dilation
    first_conv = layers.Conv1D(filters, kernel_size=conv1_kernel_size,
                               padding='valid', activation='relu', 
                               name='1st_conv')(inp)
    
    # 6 dilated convolutions with resnet-style additions
    # each layer receives the sum of feature maps 
    # from previous two layers
    # *** on a quest to have meaninful layer names *** 
    res_layers = [(first_conv, '1st_conv')] 
                                           
    for i in range(1, num_dilation_layers + 1):
        if i == 1:
            res_layers_sum = first_conv
        else:
            res_layers_sum = layers.add([l for l, _ in res_layers],
                                        name='add_{}'.format(i-1))

        # dilated convolution
        conv_layer_name = 'dil_conv_{}'.format(i)
        conv_output = layers.Conv1D(filters, kernel_size=dilation_kernel_size, 
                                    padding='valid', activation='relu', 
                                    dilation_rate=2**i, 
                                    name=conv_layer_name)(res_layers_sum)

        # get shape of latest layer and crop the previous
        # layer (index == -1) in the res_layers list to that size
        conv_output_shape = int_shape(conv_output)
        cropped_layers = []
        
        lyr, name = res_layers[-1]
        lyr_shape = int_shape(lyr)
        cropsize = lyr_shape[1] // 2 - conv_output_shape[1] // 2
        lyr_name = 'crop_{}'.format(name.split('-')[0])
        cropped_layers.append(
            (layers.Cropping1D(cropsize, name=lyr_name)(lyr), lyr_name)) 
        
        # now append the current conv_output
        cropped_layers.append((conv_output, conv_layer_name))
        
        res_layers = cropped_layers

    # the final output from the 6 dilated convolutions 
    # with resnet-style connections
    combined_conv = layers.add([l for l, _ in res_layers], 
                               name='combined_conv') 

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    profile_out_prebias = layers.Conv1D(filters=num_tasks, 
                                        kernel_size=profile_kernel_size, 
                                        padding='valid', 
                                        name='profile_out_prebias')\
                                        (combined_conv)

    # Step 1.2 - Crop to match size of the required output size, a
    #            minimum difference of 346 is required between input
    # .          seq len and ouput len
    profile_out_prebias_shape = int_shape(profile_out_prebias)
    cropsize = profile_out_prebias_shape[1] // 2 - output_len // 2
    profile_out_prebias = layers.Cropping1D(
        cropsize, name='prof_out_crop2match_output')(profile_out_prebias)

    # Step 1.3 - concatenate with the control profile 
    concat_pop_bpi = layers.concatenate(
        [profile_out_prebias, bias_profile_input], 
        name="concat_with_bias_prof", axis=-1)

    # Step 1.4 - Final 1x1 convolution
    profile_out = layers.Conv1D(filters=num_tasks, kernel_size=1, 
                                name="profile_predictions")(concat_pop_bpi)

    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet 
    #            function
    # acronym - gapcc
    gap_combined_conv = layers.GlobalAveragePooling1D(
        name='gap')(combined_conv) 
    
    # Step 2.2 Concatenate the output of GAP with bias counts
    concat_gapcc_bci = layers.concatenate(
        [gap_combined_conv, bias_counts_input], name="concat_with_bias_cnts", 
        axis=-1)
    
    # Step 2.3 Dense layer to predict final counts
    count_out = layers.Dense(num_tasks, 
                             name="logcount_predictions")(concat_gapcc_bci)
  
    if use_attribution_prior:
        if attribution_prior_params is None:
            raise NoTracebackException(
                "RuntimeError: You must provide 'attribution_prior_params' "
                "dict to use attribution priors")

        if 'frquency_limit' not in attribution_prior_params:
            raise NoTracebackException(
                "KeyError: (attribution_prior_params): 'frquency_limit'")

        if 'limit_softness' not in attribution_prior_params:
            raise NoTracebackException(
                "KeyError: (attribution_prior_params): 'limit_softness'")

        if 'grad_smooth_sigma' not in attribution_prior_params:
            raise NoTracebackException(
                "KeyError: (attribution_prior_params): 'grad_smooth_sigma'")             

        # instantiate attribution prior Model with inputs and outputs
        model = AttributionPriorModel(
            attribution_prior_params['frquency_limit'],
            attribution_prior_params['limit_softness'],
            attribution_prior_params['grad_smooth_sigma'],     
            attribution_prior_params['profile_grad_loss_weight'],
            attribution_prior_params['counts_grad_loss_weight'],
            inputs=[inp, bias_counts_input, bias_profile_input],
            outputs=[profile_out, count_out])
        
    else:
        # instantiate keras Model with inputs and outputs
        model = Model(
            inputs=[inp, bias_counts_input, bias_profile_input],
            outputs=[profile_out, count_out])

    return model
