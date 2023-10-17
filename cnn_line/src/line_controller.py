#!/usr/bin/env python

import rospy, argparse
import numpy as np
import tf_conversions

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose2D

from math import sin, cos

#line_topic = '/line/x_theta'
#odom_topic = '/mobile_base_controller/odom'
#controller = 'row_control'

# Uncertainty Rate #
un1 = 0.95
un2 = 0.8
h = 0.0333

tx  = un2 * 0.16   # X offset of the camera in relation to the robot frame
tz  = un2 * 1.4    # Z offset of the camera in relation to the robot frame
psi = un2 * 0.707  # Camera pitch

# Camera Parameters #
W    = 640.0
H    = 480.0
f    = un2 * 0.008
fov  = un2 * 1.0472
rhou = (2*f/W) * np.tan(fov/2)
rhov = (2*f/H) * np.tan(fov/2)
u0   = W/2
v0   = H/2
fline1 = un1 * 554.254691 #Com incertezas
fline2 = f/rhov #Ideal

# Control Gain for IBVS #
K_row    = 1.0  # Gain for Row controller
Krho_row = 0.05 # Gain for robust part for Row Controller
K_col    = 0.15 # Gain for classic control for Column Controller
Krho_col = 0.3  # Gain for robust part for Column Controller

gamma_row = np.array([[K_row,    0.0], [0.0, K_row]   ]) # Better case for robust control for Row Controller
rho_row   = np.array([[Krho_row, 0.0], [0.0, Krho_row]]) # Matrix Gain for Robust Part

gamma_col = np.array([[K_col,    0.0], [0.0, 2*K_col] ]) # Better case for robust control for Col Controller
rho_col   = np.array([[Krho_col, 0.0], [0.0, Krho_col]]) # Matrix Gain for Robust Part

alpha = 10.5 # Proportional gain
beta  = 2.5  # Robust gain sigma
rho1  = 1.0  # Robust gain for disturbance in z1
rho2  = 6.5  # Robust gain for disturbance in z2
epslon = 10e-3 # Tolerance for controller

#Initial Conditions
V  = np.array([1.0,1.0])
#vd = 0.5
qd = np.array([-42.69,1.85,0.43]) # qd = np.array([0,0,0])
u  = np.zeros((2))
e  = np.zeros((2))

# Desired model robot #	
kappa1 = 0.6
delta1 = 0.52
previus_control = ''

def line_callback(msg):
    global lin_x, theta, previus_control
    
    # Store line information from the received message
    lin_x = msg.x
    theta = msg.theta
    
    # Get linear velocity parameter from ROS parameter server with a default value of 0.4
    vd = rospy.get_param('/linear_velocity', 0.4)
    
    # Get velocity controller type from ROS parameter server with a default value of 'row_control'
    controller = rospy.get_param('/velocity_controller', 'row_control')
    
    # Check if the control type has changed and log the change
    if previus_control != controller:
        rospy.loginfo('New controller: ' + controller)
        previus_control = controller
    
    # Perform control based on the selected controller type
    if controller == 'cascade':
        # Calculate linear and angular velocities using cascade controller
        (vel_lin, vel_ang) = cascadeController(lin_x, theta, vd)
        
        # Set linear and angular velocities in the twist message
        vel.linear.x = np.squeeze(vel_lin)
        vel.angular.z = np.squeeze(vel_ang)
        
        # Publish the calculated twist message
        pub.publish(vel)

    elif controller == 'row_control':
        # Calculate angular velocity using row control
        vel_ang = old_row_Controller(lin_x, theta)
        
        # Calculate linear velocity with a limit and set angular velocity in the twist message
        vel.linear.x = np.squeeze(np.clip(vd - np.abs(vel_ang), a_min=0, a_max=2))
        vel.angular.z = np.squeeze(vel_ang)
        
        # Publish the calculated twist message
        pub.publish(vel)
    
    #print(vel.linear.x,vel.angular.z)

def odometry_callback(msg):
    global odom
    odom = msg

def old_row_Controller(point_x, theta, lambdax = 0.5, lambdatheta = 0.5, Vconst = 0.5, Wmax = 0.5, ro = 0.707, tz = 0.95, ty = 0.16):
    """
    Row Controller for Image-Based Visual Servoing.

    Calculates the control input `w` for a robot's row (forward motion) based on the given image point and orientation.

    Parameters:
        point_x (float): The normalized horizontal image point [-1, 1] w.r.t. the width of the image.
        theta (float): The robot's orientation (angle) in radians.
        lambdax (float, optional): Proportional gain for position error. Default is 0.5.
        lambdatheta (float, optional): Proportional gain for orientation error. Default is 0.5.
        Vconst (float, optional): Constant velocity for row motion. Default is 0.5.
        Wmax (float, optional): Maximum angular velocity allowed. Default is 0.5.
        ro (float, optional): Constant used in the control law. Default is 0.707.
        tz (float, optional): Constant used in the control law. Default is 0.95.
        ty (float, optional): Constant used in the control law. Default is 0.16.

    Returns:
        w (float): Control input representing the angular velocity for the row motion.

    Note:
        - The `point_x` parameter must be normalized within the range [-1, 1], representing the horizontal image point
          with -1 being the leftmost point, 1 being the rightmost point, and 0 being the center.
        - The `theta` parameter is the orientation of the robot, given in radians.
        - The control input `w` is the calculated angular velocity for the row motion.
    """
    # Convert the given normalized image point to robot's Y and X coordinates
    # Y is set to -1, and X is the normalized image point (point_x)
    Y = -1
    X = point_x

    # Compute the first row of the interaction matrix (Lx) for row (forward) motion control
    Lx = np.array([(-sin(ro) - Y * cos(ro))/tz, 0, (X * (sin(ro)+Y*cos(ro)))/tz, X*Y, -1-X**2, Y])

    # Compute the second row of the interaction matrix (Ltheta) for row motion control
    Ltheta = np.array([(cos(ro) * cos(theta)**2)/tz, 
                    (cos(ro) * cos(theta) * sin(theta))/tz, 
                    -(cos(ro)*cos(theta) * (Y*sin(theta) + X * cos(theta)))/tz, 
                    -(Y*sin(theta) + X*cos(theta)) *cos(theta), 
                    -(Y*sin(theta) + X*cos(theta))*sin(theta), 
                    -1])

    # Stack the Lx and Ltheta matrices vertically to form the full interaction matrix Ls
    Ls = np.vstack((Lx, Ltheta))

    # Define the translational and rotational velocity vector components for the robot
    Tv = np.array([ 0, -sin(ro),  cos(ro), 0, 0, 0 ]).transpose()[:, None]
    Tw = np.array([-ty, 0, 0, 0, -cos(ro), -sin(ro)]).transpose()[:, None]

    # Compute the control input components related to translational and rotational motion
    Ar = np.matmul(Ls, Tv)
    Br = np.matmul(Ls, Tw)

    # Calculate the pseudoinverse of Br to be used in the control law
    Brp = np.linalg.pinv(Br)

    # Calculate the position and orientation errors between the robot and the desired image point
    ex = point_x
    etheta = theta

    # Create a matrix containing the position and orientation error gains
    matriz_ganho_erro = np.array([lambdax * ex, lambdatheta * etheta]).transpose()[:,None]

    # Calculate the final control input (angular velocity) using the control law
    w = - np.matmul(Brp,(matriz_ganho_erro + Ar * Vconst))

    # Ensure that the calculated angular velocity does not exceed the maximum allowed value (Wmax)
    if(abs(w) > Wmax):
        w = Wmax * np.sign(w)

    return w

def interaction_matrix(tx,tz,phi,psi,pim):
	# Image coordinate #
	X = pim[0]
	Y = pim[1]

	# Normalized coordinate #

	# Calculating the transformation matrix #
	a1 = -np.sin(psi)*(tz*np.cos(psi) + tx*np.sin(psi))-np.cos(psi)*(tx*np.cos(psi) - tz*np.sin(psi))
	Tv = np.transpose(np.array([0, -np.sin(psi), np.cos(psi), 0, 0, 0]))
	Tw = np.transpose(np.array([a1, 0, 0, 0, -np.cos(psi), -np.sin(psi)]))
	
	# Calculating theta line of Jacobian #
	Jl_theta = np.array([[(np.cos(psi)*np.cos(phi)**2)/tz, np.cos(psi)*np.cos(phi)*np.sin(phi)/tz,
    	-(np.cos(psi)*np.cos(phi))*(Y*np.sin(phi) + X*np.cos(phi))/tz, 
    	-(Y*np.sin(phi) + X*np.cos(phi))*np.cos(phi),  
    	-(Y*np.sin(phi) + X*np.cos(phi))*np.sin(phi), -1.0]])
	
	# Calculating  x y Jacobian #
	Jim = np.array([[(-np.sin(psi)- Y*np.cos(psi))/tz, 0, X*(np.sin(psi) + Y*np.cos(psi))/tz, X*Y, -1-X**2, Y],
	[0, (-np.sin(psi)- Y*np.cos(psi))/tz, Y*(np.sin(psi) + Y*np.cos(psi))/tz, 1+Y**2, -X*Y, -X]])

	# Concatenating the Jacobians #
	J = np.concatenate((Jim, Jl_theta), axis=0)
	
	# Calculating Jw and Jv #
	Lx = np.array([J[0,0],J[0,1],J[0,2],J[0,3],J[0,4],J[0,5]])
	Ly = np.array([J[1,0],J[1,1],J[1,2],J[1,3],J[1,4],J[1,5]])
	Ltheta = np.array([J[2,0],J[2,1],J[2,2],J[2,3],J[2,4],J[2,5]])
	Lrow = np.concatenate(([Lx],[Ltheta]), axis=0)
	Lcolumn = np.concatenate(([Ly],[Ltheta]), axis=0)
        
	Jvx = np.matmul(Lrow,Tv) #EQ 2.13 TESE  
	Jwx = np.matmul(Lrow,Tw)
	Jvy = np.matmul(Lcolumn,Tv) 
	Jwy = np.matmul(Lcolumn,Tw)
        
	return Jvx,Jwx,Jvy,Jwy

def row_controller  (ui, vi, phi, vd):
    global V,e
    ubard = 0 
    vbard = H - v0
    
    #Image features in pixels
    Ximd = ubard/fline1  #Pos desejada X
    Yimd = vbard/fline2  #Pos desejada Y
    pimd = np.array([Ximd,Yimd,0])
    pim = np.zeros((3))

    # Vector of image features in meters
    ubar = ui - u0
    vbar = vi - v0
    X = ubar/fline1
    Y = vbar/fline2
    pim[0] = X
    pim[1] = Y

    # Calculating the Interaction matrix
    Jvx,Jwx,Jvy,Jwy = interaction_matrix(tx,tz,phi,psi,pim) #EQ 2.13 TESE

    Jwx = np.array([ [Jwx[0]] , [Jwx[1]] ]) #Alteração de Tupla para numpy array
    Jwy = np.array([ [Jwy[0]] , [Jwy[1]] ])

    # Calculating error
    e = np.asarray([pim[0] - pimd[0], phi - pimd[2]]) #EQ 2.15 TESE
    
    # Calculo inversa da matriz
    Jwxinv = np.linalg.pinv(Jwx)

    # Robust Controller
    dpim =  np.matmul(gamma_row,e)[:,None] + np.matmul(rho_row, V)  # Robust control + (Jvx*vd)[:,None]

    # Inputs #ou output?
    ud = np.array((vd, np.squeeze(np.matmul(-Jwxinv, dpim)))) #added squeeze

    return ud

def col_controller  (ui, vi, phi, vd):
    global V,e
    if ui < 0:
        ubard = 0 - u0 
    else:
        ubard = W - u0

    vbard = H - v0
    
    Ximd = ubard/fline1
    Yimd = vbard/fline2
    pimd = np.array([Ximd,Yimd,0])
    pim = np.zeros((3))

    # Vector of image features in meters#
    ubar = ui - u0
    vbar = vi - v0
    X = ubar/fline1
    Y = vbar/fline2
    pim[0] = X
    pim[1] = Y

    # Calculating the Interaction matrix #
    Jvx,Jwx,Jvy,Jwy = interaction_matrix(tx,tz,phi,psi,pim)

    Jwx = np.array([ [Jwx[0]] , [Jwx[1]] ])
    Jwy = np.array([ [Jwy[0]] , [Jwy[1]] ])

    # Calculating error
    e = np.asarray([pim[1] - pimd[1], phi - pimd[2]])

    # Calculating the Interaction matrix
    Jwyinv = np.linalg.pinv(Jwy)

    # Robust Controller #
    dpim = np.matmul(gamma_col,e)[:,None] + (Jvy*vd)[:,None] + np.matmul(rho_col, V) # Robust control

    # Inputs #
    ud = np.array([vd, np.squeeze(np.matmul(-Jwyinv, dpim))])
    return ud

def cascadeController(lin_x, lin_theta, vd):
    global V, e
    if lin_x < 0:
        ui = 0
        vi = int(lin_x / np.sin(lin_theta))
        controller = 'col'

    elif lin_x > W:
        ui = W
        vi = int((lin_x - W*np.cos(lin_theta)) / np.sin(lin_theta))
        controller = 'col'

    else:
        ui = lin_x
        vi = int((lin_x - H*np.sin(lin_theta)) / np.cos(lin_theta))
        controller = 'row'

    # Sliding surface #
    s = e
    V = np.sign(s)*np.abs(s)**(1/2) #Super twist | gamma = 1/2 EQ: 2.25
    V = V[:,None]
    s = s[:,None]

    if controller == 'col':
        ud = col_controller  (ui, vi, lin_theta, vd)
    else:
        ud = row_controller  (ui, vi, lin_theta, vd)

    # Get pose from odometry
    _,_,robot_yaw = tf_conversions.transformations.euler_from_quaternion([
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w
                    ])
    q = np.asarray([odom.pose.pose.position.x, odom.pose.pose.position.y, robot_yaw])

    # Get speed from odometry
    dq_read = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]

    # Estimating parameters #
    v_read =  (dq_read[0]**2 + dq_read[1]**2)**0.5 + 10e-5
    kappa  =  v_read/ud[0]
    delta  = -(dq_read[1]/ (ud[0]*kappa*np.cos(q[2]))) + np.tan(q[2])
    
    dqd = np.asarray([ud[0]*kappa1*(np.cos(qd[2]) + delta1*delta*np.sin(qd[2])),
                      ud[0]*kappa1*(np.sin(qd[2]) - delta1*delta*np.cos(qd[2])),
                      kappa1*ud[1]])

    qd[0] = qd[0] + h*dqd[0] 
    qd[1] = qd[1] + h*dqd[1]
    qd[2] = qd[2] + h*dqd[2]

    e_r = q - qd

    # Chained form #
    z = np.asarray([e_r[2],
                    e_r[0]*np.cos(q[2]) + e_r[1]*np.sin(q[2]),
                    e_r[0]*np.sin(q[2]) - e_r[1]*np.cos(q[2])])
    
    sigma = 2*z[2] - z[0]*z[1]

    # Lyapunov V1 #
    V1   = (z[0]**2 + z[1]**2)/2
    beta =  2.0 + 7.5*np.exp(-100*V1)

    if(V1 > 0.0008):
        u_chained = [-alpha*z[0] - beta*z[1]*np.sign(sigma)*np.abs(sigma)**0.5 + rho1*np.sign(z[0])*np.abs(z[0])**0.5,
                     -alpha*z[1] + beta*z[0]*np.sign(sigma)*np.abs(sigma)**0.5 + rho2*np.sign(z[1])*np.abs(z[1])**0.5]
    else:
        u_chained = [-beta*z[1]*np.sign(sigma)*np.abs(sigma)**0.5,
                     +beta*z[0]*np.sign(sigma)*np.abs(sigma)**0.5]
    
    #print(u_chained,ud[0],np.cos(z[0]),u[1],z[2])

    u[1] = u_chained[0] + ud[1]
    u[0] = u_chained[1] + ud[0]*np.cos(z[0]) + u[1]*z[2]

    vel_lin = kappa1*ud[0]
    vel_ang = kappa1*ud[1]

    #print(u,ud)

    return (vel_lin, vel_ang)

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('line_controller')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--line_topic',   type=str, nargs='?', default='/line/x_theta')
    parser.add_argument('--odom_topic',   type=str, nargs='?', default='/mobile_base_controller/odom')
    #parser.add_argument('--controller',   type=str, nargs='?', default='row_control')
    parser.add_argument('--output_topic', type=str, nargs='?', default='/mobile_base_controller/cmd_vel')
    parser.add_argument('__name', type=str, nargs='?', default='')
    parser.add_argument('__log', type=str, nargs='?', default='')
    args = parser.parse_args()
    
    line_topic = args.line_topic
    odom_topic = args.odom_topic
    #controller = args.controller
    output_topic = args.output_topic

    # Initialize subscribers and publisher
    odom = Odometry()
    vel = Twist()
    lin_x = 0
    theta = 0

    rospy.Subscriber(line_topic , Pose2D  , line_callback)
    rospy.Subscriber(odom_topic , Odometry, odometry_callback)
    pub = rospy.Publisher(output_topic, Twist, queue_size=10)

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass