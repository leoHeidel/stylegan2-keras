import tensorflow as tf

import style_gan_3d

def cross(vector1, vector2):
    """
    3D cross product between two vectors, potentially batched
    """
    vector1_x, vector1_y, vector1_z = tf.unstack(vector1, axis=-1)
    vector2_x, vector2_y, vector2_z = tf.unstack(vector2, axis=-1)
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return tf.stack((n_x, n_y, n_z), axis=-1)

def trace_ray(camera_position, camera_direction, camera_rotation, side, side_count=8):
    """
    Output the positions to be sampled, as a 3d map
    works on batches 
    
    camera_position: 3-d vector
    camera_direction: 3-d unite vector
    camera_rotaion: float, radian, zero is default
    side: float, radian, what angle the camera is able to view
    """    
    #Finding camera orientation before rotation
    y_dir = tf.constant([0,1,0], dtype=tf.float32) #(*,3)
    camera_x = cross(y_dir, camera_direction) #(*,3)
    camera_y = cross(camera_direction, camera_x) #(*,3)
    
    #Applying camera rotation
    rot_cos = tf.cos(camera_rotation[:,tf.newaxis]) # (*,)
    rot_sin = tf.sin(camera_rotation[:,tf.newaxis]) # (*,)
    camera_x, camera_y = (rot_cos*camera_x + rot_sin*camera_y, 
                          -rot_sin*camera_x + rot_cos*camera_y) # (*,3)
        
    dist_to_zero = tf.norm(camera_position, axis=-1) #(*,)
    half_side = side*dist_to_zero/2 # (*,)
    linspace = tf.linspace(-half_side,half_side, side_count, axis=1) #(*,side_count)
    #Relative to the center of the grid
    grid_x = camera_x[:,tf.newaxis,tf.newaxis,:] * linspace[:,tf.newaxis,:,tf.newaxis] #(*,1,side_count,3)
    grid_y = - camera_y[:,tf.newaxis,tf.newaxis,:] * linspace[:,:,tf.newaxis,tf.newaxis] #(*,side_count,1,3)
    
    grid_xy = grid_x + grid_y # (*,side_count,side_count,3)
    #Relative to camera 
    grid_xy = grid_xy + (dist_to_zero[:,tf.newaxis]*camera_direction)[:,tf.newaxis,tf.newaxis,:]
    
    grid_xyz = grid_xy[:,tf.newaxis,:,:,:] * (1 + linspace/dist_to_zero[:,tf.newaxis])[:,:,tf.newaxis,tf.newaxis,tf.newaxis] # (*,side_count,side_count,side_count,3)    
    return grid_xyz + camera_position[:,tf.newaxis,tf.newaxis,tf.newaxis,:]

def to_feature_map(feature_3D):
    """
    Sums the rays along axis 1, output a feature map 
    """
    coefficients = tf.reduce_sum(tf.abs(feature_3D),axis=4, keepdims=True) #(*,side_count,side_count,side_count,1)
    coefficients = 1/(1 + tf.math.cumsum(coefficients, axis=1)) #(*,side_count,side_count,side_count,1)
    feature_map = tf.reduce_sum(coefficients*feature_3D, axis=1) #(*,side_count,side_count,1)
    return feature_map