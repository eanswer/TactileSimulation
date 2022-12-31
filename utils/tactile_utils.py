import numpy as np
import cv2

def visualize_tactile_image(tactile_array, tactile_resolution = 30, shear_force_threshold = 0.00015, normal_force_threshold = 0.0008):
    resolution = tactile_resolution
    T = len(tactile_array)
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype = float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_array[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_array[row, col][1] / shear_force_threshold * resolution
            color = (0., max(0., 1. + tactile_array[row][col][2] / normal_force_threshold), min(1., -tactile_array[row][col][2] / normal_force_threshold))
            
            cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color, 6, tipLength = 0.4)
    
    return imgs_tactile

def visualize_depth_image(tactile_forces, threshold = 0.0012):
    img = np.zeros((tactile_forces.shape[0], tactile_forces.shape[1]))
    for i in range(tactile_forces.shape[0]):
        for j in range(tactile_forces.shape[1]):
            img[i][j] = min(1., -tactile_forces[i][j][2] / threshold)
    return img