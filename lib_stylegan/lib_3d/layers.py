import tensorflow as tf
import tensorflow.keras as keras

from . import math_3d 



class CameraStd(keras.layers.Layer):
    def __init__(self):
        super(CameraStd, self).__init__()
        """
        Learnable parameters for the standard deviations of the random gaussians noise
        """
        self.theta_std = self.add_weight(name="theta_std", initializer=tf.constant_initializer(0.2))
        self.phi_std = self.add_weight(name="phi_std", initializer=tf.constant_initializer(1.))
        self.rotation_std = self.add_weight(name="rotation_std", initializer=tf.constant_initializer(0.1))
        self.center_std = self.add_weight(name="center_std", initializer=tf.constant_initializer(0.1))
        self.side_std = self.add_weight(name="side_std", initializer=tf.constant_initializer(0.1))
        self.distance_std = self.add_weight(name="distance_std", initializer=tf.constant_initializer(4))
        self.base_distance = self.add_weight(name="base_distance", initializer=tf.constant_initializer(15))
    
        self.dir_x = tf.constant([1,0,0], dtype=tf.float32)
        self.dir_y = tf.constant([0,1,0], dtype=tf.float32)
        self.dir_z = tf.constant([0,0,1], dtype=tf.float32)
    
    def call(self, inputs):
        """
        Inputs, all random values zero centered std 1:
        camera_angles : (*,3)
        center : (*,3)
        distance : (*)
        side : (*)
        """
        camera_angles, center, distance, side = inputs
        
        #Rotation
        theta = camera_angles[:,0] * self.theta_std # (*,)
        phi = camera_angles[:,1] * self.phi_std # (*,)
        camera_rotation = camera_angles[:,2] * self.rotation_std # (*,)
        #print(f"theta: {theta.numpy()}")
        #print(f"phi: {phi.numpy()}")
        #print(f"camera_rotation: {camera_rotation.numpy()}")
        
        #Camera position
        distance = self.base_distance+self.distance_std*distance
        #print(f"distance: {distance.numpy()}")
        dir_camera_phi = tf.cos(phi[:,tf.newaxis]) * self.dir_z + tf.sin(phi[:,tf.newaxis]) * self.dir_x 
        dir_camera = tf.cos(theta[:,tf.newaxis]) * dir_camera_phi + tf.sin(theta[:,tf.newaxis]) * self.dir_y # (*, 3)
        camera_position = dir_camera * distance[:,tf.newaxis] # (*, 3)
            
        #Camera direction
        center = center * self.center_std # (*, 3)
        camera_direction, _ = tf.linalg.normalize(center - camera_position, axis=1)  # (*, 3)
        
        #Side
        side = (self.side_std*side + 1) / distance # (*,)
            
        return camera_position, camera_direction, camera_rotation, side
    
class RayTracer(keras.layers.Layer):
    def __init__(self, side_count=8):
        """
        Sample coordinate accross rays.
        output is of shape (*,side_count,side_count,side_count,3)
        """
        super(RayTracer, self).__init__()
        
        self.side_count = side_count

    def call(self, inputs):
        camera_position, camera_direction, camera_rotation, side = inputs 
        return math_3d.trace_ray(camera_position, camera_direction, camera_rotation, side, side_count=self.side_count)
    
