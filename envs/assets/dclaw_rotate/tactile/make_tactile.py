import numpy as np
import trimesh

    
if __name__ == "__main__":
    obj_path = 'two3_link.obj'
    
    mesh = trimesh.load_mesh(obj_path, file_type = 'obj')

    fp = open('dclaw_fingertip_tactile.txt', 'w')
    
    tactile_sensors = []
    center = np.array([0., 0.07372, 0.])
    for i in range(mesh.vertices.shape[0]):
        if mesh.vertices[i][1] > 0.07:
            pos = mesh.vertices[i]
            image_pos = (np.array([mesh.vertices[i][0], mesh.vertices[i][2]]) + 0.01) * 1000
            normal = np.array([0., 1., 0.])
            axis_0 = np.array([1., 0., 0.])
            axis_1 = np.array([0., 0., 1.])
            tactile_sensors.append({'pos': pos, 'image_pos': image_pos, 'normal': normal, 'axis_0': axis_0, 'axis_1': axis_1})
    
    fp.write(f'{len(tactile_sensors)}\n')
    for i in range(len(tactile_sensors)):
        pos = tactile_sensors[i]['pos']
        image_pos = tactile_sensors[i]['image_pos']
        normal = tactile_sensors[i]['normal']
        axis_0 = tactile_sensors[i]['axis_0']
        axis_1 = tactile_sensors[i]['axis_1']
        fp.write(f'"{pos[0]} {pos[1]} {pos[2]}" "{int(image_pos[0])} {int(image_pos[1])}" "{normal[0]} {normal[1]} {normal[2]}" "{axis_0[0]} {axis_0[1]} {axis_0[2]}" "{axis_1[0]} {axis_1[1]} {axis_1[2]}"\n')
    
    fp.close()

