
#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf    
from igm.utils.math.interpolate_bilinear_tf import interpolate_bilinear_tf

def interpolate_particles_2d(U, V, W, SMB, THK, TOPG, indices):

    u = interpolate_bilinear_tf(U[..., None], indices, indexing="ij")[:, :, 0]
    v = interpolate_bilinear_tf(V[..., None], indices, indexing="ij")[:, :, 0]
    w = interpolate_bilinear_tf(W[..., None], indices, indexing="ij")[:, :, 0]
    smb = interpolate_bilinear_tf(SMB[None, ..., None], indices, indexing="ij")[0, :, 0]
    thk = interpolate_bilinear_tf(THK[None, ..., None], indices, indexing="ij")[0, :, 0]
    topg = interpolate_bilinear_tf(TOPG[None, ..., None], indices, indexing="ij")[0, :, 0] 

    return u, v, w, smb, thk, topg

