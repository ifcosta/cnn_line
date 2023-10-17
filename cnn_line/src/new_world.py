#!/usr/bin/python3
import os, rospkg
import numpy as np

def rng(mu,sig):
    s = np.random.normal(mu, sig/3, 1)
    return np.squeeze(s)

def place_model(idx,plant_type,x_pos,y_pos,z_pos,z_rot):
    print(f'    <model name="plt_{str(idx)}">',file = output)
    print(f'      <pose> {x_pos:.3} {y_pos:.3} {z_pos:.3} 0.0 0.0 {z_rot:.5} </pose>',file = output)
    print(f'      <static>true</static>',file = output)
    print(f'      <link name="body">',file = output)
    print(f'        <visual name="visual">',file = output)
    print(f'        <cast_shadows>1</cast_shadows>',file = output)
    print(f'          <geometry>',file = output)
    print(f'            <mesh><uri>file://{package_path}/worlds/models/plant{str(plant_type)}.obj</uri></mesh>',file = output)
    print(f'          </geometry>',file = output)
    print(f'        </visual>',file = output)
    print(f'      </link>',file = output)
    print(f'    </model>',file = output)

def place_slope(idx,slope_type,x_pos,y_pos,z_pos,z_rot):
    print(f'    <model name="slope_{str(idx)}">',file = output)
    print(f'      <pose> {x_pos:.3} {y_pos:.3} {z_pos:.3} 0.0 0.0 {z_rot:.5} </pose>',file = output)
    print(f'      <static>true</static>',file = output)
    print(f'      <link name="body">',file = output)
    print(f'        <visual name="visual">',file = output)
    print(f'        <cast_shadows>1</cast_shadows>',file = output)
    print(f'          <geometry>',file = output)
    print(f'            <mesh><uri>file://{package_path}/worlds/models/slope.obj</uri></mesh>',file = output)
    print(f'          </geometry>',file = output)
    print(f'        </visual>',file = output)
    print(f'      </link>',file = output)
    print(f'    </model>',file = output)

rospack = rospkg.RosPack()

package_path = rospack.get_path('cnn_line')

input = open(os.path.join(package_path, 'worlds/field_gen.world'), 'r')
world = input.readlines()
input.close()

output = open(os.path.join(package_path, 'worlds/field_gen.world'), 'w')

new_workspace = package_path.split('/')[-3]
new_name = package_path.split('/')[-4]
old_workspace = world[2][4:-4]
old_name = world[3][4:-4]

for lines in world[:112]:
    print(lines.rstrip().replace(old_workspace,new_workspace).replace(old_name,new_name),file = output)
    pass

rows = np.asarray([ -1.25, -0.75 ,-0.25 , 0.25 , 0.75 , 1.25])
x_pos = np.arange(0 , 10.1 , 0.3)
y_pos = np.zeros_like(x_pos)
z_rot = np.zeros_like(x_pos)

idx = 0
for row in rows:
    for plant in range(len(y_pos)):
        x   = (0.0 + x_pos[plant] + rng(0,0.1)) * (1)
        y   =  0.0 + y_pos[plant] + rng(0,0.075) + row
        z   = rng(0.005, 0.005) + 0.01

        if np.random.random() > 0.15:
            place_model(idx      , np.random.randint( 0, 8) , x_pos=x , y_pos=y -  0, z_pos = z , z_rot = z_rot[plant] + rng(0,np.pi))
            place_model(idx+10000, np.random.randint(10,14) , x_pos=x , y_pos=y -  3, z_pos = z , z_rot = z_rot[plant] + rng(0,np.pi))
            place_model(idx+20000, np.random.randint(20,23) , x_pos=x , y_pos=y -  6, z_pos = z , z_rot = z_rot[plant] + rng(0,np.pi))
            place_model(idx+30000, np.random.randint(30,37) , x_pos=x , y_pos=y -  9, z_pos = z , z_rot = z_rot[plant] + rng(0,np.pi))
            place_model(idx+40000, 37 , x_pos=x , y_pos=y -  12, z_pos = z , z_rot = z_rot[plant] + rng(0,np.pi))
        idx += 1

    place_slope(idx     , 0 , x_pos=8.0 , y_pos=-0.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+ 100, 0 , x_pos=8.0 , y_pos=-0.0+row , z_pos = -0.005 , z_rot=np.pi)
    place_slope(idx+ 200, 0 , x_pos=2.0 , y_pos=-0.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+ 300, 0 , x_pos=2.0 , y_pos=-0.0+row , z_pos = -0.005 , z_rot=np.pi)

    place_slope(idx+ 400, 0 , x_pos=8.0 , y_pos=-3.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+ 500, 0 , x_pos=8.0 , y_pos=-3.0+row , z_pos = -0.005 , z_rot=np.pi)
    place_slope(idx+ 600, 0 , x_pos=2.0 , y_pos=-3.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+ 700, 0 , x_pos=2.0 , y_pos=-3.0+row , z_pos = -0.005 , z_rot=np.pi)

    place_slope(idx+ 800, 0 , x_pos=8.0 , y_pos=-6.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+ 900, 0 , x_pos=8.0 , y_pos=-6.0+row , z_pos = -0.005 , z_rot=np.pi)
    place_slope(idx+1000, 0 , x_pos=2.0 , y_pos=-6.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+1100, 0 , x_pos=2.0 , y_pos=-6.0+row , z_pos = -0.005 , z_rot=np.pi)

    place_slope(idx+1200, 0 , x_pos=8.0 , y_pos=-9.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+1300, 0 , x_pos=8.0 , y_pos=-9.0+row , z_pos = -0.005 , z_rot=np.pi)
    place_slope(idx+1400, 0 , x_pos=2.0 , y_pos=-9.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+1500, 0 , x_pos=2.0 , y_pos=-9.0+row , z_pos = -0.005 , z_rot=np.pi)

    place_slope(idx+1600, 0 , x_pos=8.0 , y_pos=-12.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+1700, 0 , x_pos=8.0 , y_pos=-12.0+row , z_pos = -0.005 , z_rot=np.pi)
    place_slope(idx+1800, 0 , x_pos=2.0 , y_pos=-12.0+row , z_pos = -0.005 , z_rot=  0.0)
    place_slope(idx+1900, 0 , x_pos=2.0 , y_pos=-12.0+row , z_pos = -0.005 , z_rot=np.pi)


for lines in world[-54:]:
    print(lines.rstrip().replace(old_workspace,new_workspace),file = output)

print("world generated")

output.close()




