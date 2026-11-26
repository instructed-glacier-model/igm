#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
Utility functions for mapping interfaces.
"""

import warnings
import numpy as np


def process_inputs_scales(inputs_scales, inputs_list):
    """
    Convert inputs_scales dictionary to a numpy array in the order of inputs_list.
    
    Args:
        inputs_scales: Dict mapping field names to scales
        inputs_list: List of input field names in order
        
    Returns:
        numpy array of scales in the same order as inputs_list
    """
    scales_array = []
    for field_name in inputs_list:
        if field_name in inputs_scales:
            scales_array.append(inputs_scales[field_name])
        else:
            # Default to 1.0 if not specified
            scales_array.append(1.0)
            warnings.warn(f"Scale not specified for field '{field_name}', using default value 1.0")
    return np.array(scales_array)

def process_inputs_variances(inputs_variances, inputs_list):
    """
    Convert inputs_variances dictionary to a numpy array in the order of inputs_list.
    
    Args:
        inputs_variances: Dict mapping field names to variances
        inputs_list: List of input field names in order
        
    Returns:
        numpy array of variances in the same order as inputs_list
    """
    variances_array = []
    for field_name in inputs_list:
        if field_name in inputs_variances:
            variances_array.append(inputs_variances[field_name])
        else:
            # Default to 1.0 if not specified
            variances_array.append(1.0)
            warnings.warn(f"Scale not specified for field '{field_name}', using default value 1.0")
    return np.array(variances_array)