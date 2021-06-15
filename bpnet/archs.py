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

def TaskModule(conv_module_output, conv_module_output_pooled, 
               bias_profile_input, bias_counts_input, task_id):
    """
    
    """
    
    # Profile output
    # Step 1 - concatenate with bias profile input
    if bias_profile_input is not None:
        concat_with_bias_profile_input = layers.concatenate(
            [conv_module_output, bias_profile_input], 
            name="concat_with_bias_prof_task_{}".format(task_id), axis=-1)
    else:
        concat_with_bias_profile_input = conv_module_output
    
    # Step 2 - 1x1 convolution to yield the profile output prediction
    # for this task
    one_by_one_conv = layers.Conv1D(
        filters=1, kernel_size=1, 
        name="profile_predictions_{}".format(
            task_id))(concat_with_bias_profile_input)
    
    # Logcounts output
    # Step 1 - concatenate with bias counts input
    if bias_counts_input is not None:
        concat_with_bias_counts_input = layers.concatenate(
            [conv_module_output_pooled, bias_counts_input],
            name="concat_with_bias_counts_{}".format(task_id), axis=-1)
    else:
        concat_with_bias_counts_input = conv_module_output_pooled
        
    # Step 2 - Dense layer to yield logcounts prediction for this task
    dense = layers.Dense(
        1, name="logcount_predictions_{}".format(
            task_id))(concat_with_bias_counts_input)

    return (one_by_one_conv, dense)


def get_detailed_tasks_info(tasks):
    """
    
    """
    
    # number of input tasks
    num_tasks = len(list(tasks.keys()))
    
    # maintain a list of start and end indices to reference into
    # bias_profile_input to retrieve the corresponding bias tracks
    bias_profile_input_start_end_indices = []
    
    # track total bias tracks as you process each task's info
    total_bias_tracks = 0
    
    for i in range(num_tasks):
        start_idx = total_bias_tracks
        
        # the number of original bias tracks (non-smoothed) for this
        # task
        num_bias_tracks = len(tasks[i]['bias'])
        
        # if no bias tracks are found for this task
        if num_bias_tracks == 0:
            bias_profile_input_start_end_indices.append(None)
            continue

        # update total
        total_bias_tracks += num_bias_tracks
        
        # the length of the 'bias_smoothing' list should be the same
        # as the 'bias' list
        if len(tasks[i]['bias_smoothing']) != num_bias_tracks:
            raise NoTracebackException(
                "RuntimeError ('bias_smoothing'): Length mismatch with 'bias'")
        
        # count the number of 'smoothed' bias tracks that will be
        # added on
        for j in range(num_bias_tracks):
            if tasks[i]['bias_smoothing'][j] is not None:
                # add 1 for every smoothed track, not all bias tracks
                # may have their corresponding smoothed versions
                total_bias_tracks += 1
    
        end_idx = total_bias_tracks
        bias_profile_input_start_end_indices.append([start_idx, end_idx])
    
    return num_tasks, total_bias_tracks, bias_profile_input_start_end_indices

def BPNet(
    tasks, input_seq_len=2114, output_len=1000, filters=64, 
    num_dilation_layers=8, conv1_kernel_size=21, dilation_kernel_size=3, 
    prebias_profile_kernel_size=75, profile_kernel_size=1,
    use_attribution_prior=False, attribution_prior_params=None):
    
    """
        BPNet model architecture as described in the BPNet paper
        https://www.biorxiv.org/content/10.1101/737981v1.full.pdf
        
        Args:
            input_seq_len (int): length of input DNA sequence
            
            output_len (int): length of the profile output
                        
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
            
            num_bias_tracks_per_task (int): list of number of 
                control/bias tracks for each task

            smooth_bias_tracks (boolean): Nested list of boolean
                values to indicate if each bias track has a smoothed
                version included
            
            use_attribution_prior (boolean): indicate whether to use 
                attribution prior model
                
            attribution_prior_params (dict): python dictionary with 
                keys 'frequency_limit', 'limit_softness' & 
                'smooth_sigma'
            
        Returns:
            keras.model.Model
        
    """

    num_tasks, total_bias_tracks, indices = get_detailed_tasks_info(tasks)

    print("Total bias tracks", num_tasks)
    print("Start end indices", indices)
    # The three inputs to BPNet
    inp = layers.Input(shape=(input_seq_len, 4), name='sequence')
    
    bias_counts_input = layers.Input(
        shape=(1, num_tasks), name="bias_logcounts")
    
    bias_profile_input = layers.Input(
        shape=(output_len, total_bias_tracks), name="bias_profiles")
    # end inputs

    print("bias_profiles", bias_profile_input.shape)
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

    # the final output from the dilated convolutions with 
    # resnet-style connections
    dilation_layers_out = layers.add([l for l, _ in res_layers], 
                               name='dilation_layers_out') 

    # Branch 1. Profile prediction
    # Step 1.1 - 1D convolution with a very large kernel
    profile_out_prebias = layers.Conv1D(
        filters=num_tasks, kernel_size=prebias_profile_kernel_size, 
        padding='valid', name='profile_out_prebias')(dilation_layers_out)

    # Step 1.2 - Crop to match size of the required output size, a
    #            minimum difference of 346 is required between input
    # .          seq len and ouput len
    profile_out_prebias_shape = int_shape(profile_out_prebias)
    cropsize = profile_out_prebias_shape[1] // 2 - output_len // 2
    profile_out_prebias = layers.Cropping1D(
        cropsize, name='prof_out_crop2match_output')(profile_out_prebias)

    
    # Branch 2. Counts prediction
    # Step 2.1 - Global average pooling along the "length", the result
    #            size is same as "filters" parameter to the BPNet 
    #            function
    # acronym - gapcc
    gap_combined_conv = layers.GlobalAveragePooling1D(
        name='gap')(combined_conv) 
    
    # invoke task module for each task
    profile_outputs = []
    counts_outputs = []
    for i in range(num_tasks):
        _bias_profile_input = None
        _bias_counts_input = None
        if indices[i] is not None:
            # get the slice of the bias profile input for this task
            _bias_profile_input = layers.Lambda(
                lambda x: x[...,indices[i][0]:indices[i][1]], 
                name="bias_prof_{}".format(i))(bias_profile_input)
        
            # get the slice of bias counts input specific to this task
            _bias_counts_input = layers.Lambda(
                lambda x: x[..., i], 
                name="bias_counts_{}".format(i))(bias_counts_input)
        
        (one_by_one_conv, dense) = TaskModule(
            profile_out_prebias, gap_combined_conv, _bias_profile_input, 
            _bias_counts_input, i)
        profile_outputs.append(one_by_one_conv)
        counts_outputs.append(dense)

    # Concatenate all the per task profile outputs
    profile_output = layers.concatenate(
        profile_outputs, name="profile_predictions", axis=-1)
                              
    # Concatenate all the per task counts outputs
    counts_output = layers.concatenate(
        counts_outputs, name="logcount_predictions", axis=-1)
            
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
            outputs=[profile_output, counts_output])

    return model
