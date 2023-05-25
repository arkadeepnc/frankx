from frankx import Affine, Kinematics, NullSpaceHandling
from frankx import JointMotion, LinearMotion, Robot, RealtimeConfig
import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import pdb



def goToPose(robot, goal_in_world_coord, pose_in_world_coord):
     #  sending robot to a goal 
    state = robot.read_once()
    current_joints = state.q
    current_elbow = state.elbow

    # Forward kinematic
    # x = Affine(Kinematics.forward(current_joints))
    # print('Current end effector position: ', x)

    # Define new target position
    x_new = Affine(x=goal_in_world_coord[0], y=goal_in_world_coord[1], z=goal_in_world_coord[2],\
                   a = pose_in_world_coord[2],  b= pose_in_world_coord[0], c = pose_in_world_coord[1]) # Eigen Affine does Z-X-Y rotations
     
    # check feasibility of pose before it triggers collision warning

    # Franka has 7 DoFs, so what to do with the remaining Null space?
    # null_space = NullSpaceHandling(3, 0.) # Set elbow joint to 1.4

    # Inverse kinematic with target, initial joint angles, and Null space configuration
    q_new = Kinematics.inverse(x_new.vector(), current_joints, None)

    ret = robot.move(JointMotion(q_new))
    if not ret:
        print(ret,'<---- ret')
        exit(0)
    else:
        print("Success: {}".format(ret))
        pass

def invPoseSE3(T):
    assert T.shape == (4,4)
    T_inv = np.eye(4)
    R = T[0:3, 0:3]
    R_inv = R.T
    t = T[0:3, -1].reshape(3,1)
    t_inv = -R_inv@t
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, -1] = t_inv.reshape(3,)
    return T_inv

def poseVecToTmat(posevec):
    # make T mat from posevec
    r_rotvec = R.from_rotvec([posevec[3], posevec[4], posevec[5]])
    # r_rotvec = R.from_quat([posevec[3], posevec[4], posevec[5], posevec[6]])
    r_rotmat = r_rotvec.as_matrix()
    T = np.eye(4)
    T[0:3,0:3] = r_rotmat
    T[0, 3] = posevec[0]
    T[1, 3] = posevec[1]
    T[2, 3] = posevec[2]
    return T

def makeSkew(a):
    a = a.reshape(3,)
    skw = np.zeros((3,3)) 
    skw[0,1] = -a[2]
    skw[0,2] = a[1]
    skw[1,0] = a[2]
    skw[1,2] = -a[0]
    skw[2,0] = -a[1]
    skw[2,1] = a[0]
    return skw

def getRotationMtxRodrigues(a, b):
    """https://math.stackexchange.com/a/2672702"""
    a = a.reshape(3,)
    b = b.reshape(3,)
    a_cross_b = np.cross(a,b)
    a_dot_b = np.dot(a,b)
    R = 1./(np.linalg.norm(a) * np.linalg.norm(b)) * a_dot_b * np.eye(3) + \
         makeSkew(a_cross_b) + \
        ((np.linalg.norm(a) * np.linalg.norm(b)) - a_dot_b)/(np.linalg.norm(a_cross_b) + 1e-7)**2 * a_cross_b.reshape(3,1) @ a_cross_b.reshape(1,3)
    return R

def getRvecTvecFromSE3(T):
    ''' returns rvec, tvec from T'''
    R = T[0:3, 0:3]
    rvec = cv2.Rodrigues(R)[0]
    tvec = T[0:3, -1]
    return rvec.reshape(3,), tvec

def plotFrames(list_of_frames, name_of_frames, scale_fac = 1., ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot((111),aspect='auto',projection='3d')
    uo=np.array([1,0,0])  # x axis
    vo=np.array([0,1,0])  # y axis
    wo=np.array([0,0,1])  # z axis
    ax.set_xlim3d(0.* scale_fac, scale_fac)
    ax.set_ylim3d(- scale_fac/2,  scale_fac/2)
    ax.set_zlim3d(0.* scale_fac, scale_fac)
    # ax.quiver(0, 0, 0, uo, vo, wo, length=1, normalize=True)
    ax.quiver(0, 0, 0,uo[0], uo[1],uo[2], color ='r', length=0.2*scale_fac, normalize=True)
    ax.quiver(0, 0, 0,vo[0], vo[1],vo[2], color ='g', length=0.2*scale_fac, normalize=True)
    ax.quiver(0, 0, 0,wo[0], wo[1],wo[2], color ='b', length=0.2*scale_fac, normalize=True)
    ax.scatter(0., 0.,0., marker = "$"+'base'+"$" , s=1500)
    if name_of_frames is not None:
        for frame, name in zip(list_of_frames, name_of_frames):
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,0], frame[1,0],frame[2,0], color ='r', length=0.1*scale_fac, normalize=True)
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,1], frame[1,1],frame[2,1], color ='g', length=0.1*scale_fac, normalize=True)
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,2], frame[1,2],frame[2,2], color ='b', length=0.1*scale_fac, normalize=True)
            ax.scatter(frame[0,-1], frame[1,-1], frame[2, -1], marker = "$"+ name +"$" , s=1500)
    else:
        for i, frame in enumerate(list_of_frames):
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,0], frame[1,0],frame[2,0], color ='r', length=0.1*scale_fac, normalize=True)
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,1], frame[1,1],frame[2,1], color ='g', length=0.1*scale_fac, normalize=True)
            ax.quiver(frame[0,-1], frame[1,-1], frame[2, -1],frame[0,2], frame[1,2],frame[2,2], color ='b', length=0.1*scale_fac, normalize=True)
            ax.scatter(frame[0,-1], frame[1,-1], frame[2, -1], marker = "$"+ str(i) +"$" , s=100)


    return ax

def getEEFPose(robot):
    # returns robot eef pose after correcting for the ISO mount rotation
    # robot_pose = robot.current_pose()
    state = robot.read_once()
    T_base_to_eef_rotated = np.asarray(state.O_T_EE).reshape(4,4).T
    # print(T_base_to_eef_rotated.reshape(4,4))
    # exit()
    # joints = state.q
    # robot_pose = Affine(Kinematics.forward([joints[0], joints[1], joints[2],joints[3],joints[4],joints[5], joints[6]]))
    # for i, _ in enumerate(joints):
    #     print(f'joint {i} value: {_}') 

    
    # T_base_to_eef_rotated = poseVecToTmat(np.array([robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.q_w , robot_pose.q_x, robot_pose.q_y, robot_pose.q_z]))
    # z_axis = T_base_to_eef_rotated[0:3,2] 
    # angle =  np.pi/4 
    # z_axis = z_axis / np.linalg.norm(z_axis)
    # ax_ang = z_axis * angle
    # rotation_ = cv2.Rodrigues(ax_ang)[0]
    # correction_t_mat = np.eye(4)
    # correction_t_mat[0:3,0:3] = rotation_
    return T_base_to_eef_rotated  #@ correction_t_mat



def moveCameras(obj_center, radius = 0.25, phi = 45, theta = 45, theta_min = -90, num_phi = 5, num_theta = 5):
    phi*=np.pi/180
    theta*=np.pi/180
    theta_min*=np.pi/180
    # board_ring, ring_pin_IDs, ring_frame_names = initLightRing('/dev/ttyACM0')

    # cam = stereoCamera(frame_size=(768,768))

    phi_steps = np.linspace(-phi/2., phi/2., num=num_phi)
    # theta_steps = np.linspace(-theta/2., theta/2., num=num_theta)
    theta_steps = np.linspace(theta_min, theta_min+theta, num=num_theta)

    theta_mgrid, phi_mgrid = np.meshgrid(theta_steps, phi_steps)
    nx = np.sin(theta_mgrid.flatten()) * np.cos(phi_mgrid.flatten())
    ny = np.sin(theta_mgrid.flatten()) * np.sin(phi_mgrid.flatten())
    nz = np.cos(theta_mgrid.flatten())

    x = obj_center[0] + radius * nx
    y = obj_center[1] + radius * ny 
    z = obj_center[2] + radius * nz 

    normal_slopes = np.stack([nx, ny, nz], axis = -1)
    # print(normal_slopes.shape)
    rotate_pi_x_axis = np.array([[ 1.0,  0.0,  0.0],[0.0, -1.0, 0.0],[0.0,  0.0, -1.0 ]])
    rotate_pi_y_axis_3x3 = np.array([[ -1.0,  0.0,  0.0], [0.0,  1.0,  0.0], [-0.0,  0.0, -1.0]])
    rotate_pi_y_axis_4x4 = np.array([[ -1.0,  0.0,  0.0, 0.], [0.0,  1.0,  0.0, 0.],\
                                      [-0.0,  0.0, -1.0, 0], [0.,0.,0.,1]])
    rotate_pi_by_2_z_axis_3x3 = np.array([[ 0.0, 1.0,  0.0], [-1.0,  0.0,  0.0],[0.0,  0.0,  1.0]])
    rotate_pi_by_2_z_axis_4x4 = np.array([[ 0.0, 1.0,  0.0, 0], [-1.0,  0.0,  0.0, 0.],[0.0,  0.0,  1.0, 0], [0.,0.,0.,1. ]])
    rotate_pi_z_axis = np.array([[ -1.0, -0.0,  0.0, 0],[0.0, -1.0,  0.0, 0], [0.0,  0.0,  1.0, 0 ],[0.,0.,0.,1.]])
    flip_Z_and_X = np.array([[-1.,0.0, 0., 0.],[0., 1., 0., 0.],[0.,0.,-1., 0],[0.,0.,0.,1]])
    # undo_ISO_6_mount_rotation = np.array([[0.7071068, -0.7071068, 0., 0.],[0.7071068, 0.7071068, 0., 0.],[0.,0.,1., 0],[0.,0.,0.,1]])
    # undo_ISO_6_mount_rotation = np.array([[1, 0, 0., 0.],[0., 0.7071068, -0.7071068, 0.],[0.,0.7071068,0.7071068, 0],[0.,0.,0.,1]])

    T_mats = []
    for i in range(normal_slopes.shape[0]):
        r_mat =  getRotationMtxRodrigues(np.array([0.,0.,1.]), normal_slopes[i,:]) @ rotate_pi_y_axis_3x3 @ rotate_pi_by_2_z_axis_3x3
        t_mat = np.eye(4)
        t_mat[0:3,0:3] = r_mat
        t_mat[0,-1] = x[i]
        t_mat[1,-1] = y[i]
        t_mat[2,-1] = z[i]
        T_mats.append(t_mat)
        # T_eef_to_cam = np.eye(4)

    # ax = plotFrames(T_mats, None)
    # ax.scatter(x, y, z, marker='.')
    # ax.scatter(obj_center[0], obj_center[1], obj_center[2], marker='*')
    # plt.show()
    # get robot's current pose
    # pdb.set_trace()
    # robot_pose = robot.current_pose()
    T_base_to_eef = getEEFPose(robot)
    # T_base_to_eef = poseVecToTmat(np.array([robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.q_w , robot_pose.q_x, robot_pose.q_y, robot_pose.q_z]))
    # T_base_to_eef = poseVecToTmat(np.array([robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.b , robot_pose.c, robot_pose.a])) # @ flip_Z_and_X  #@ rotate_pi_y_axis_4x4 #@ rotate_pi_by_2_z_axis_4x4
    ax = plotFrames([T_base_to_eef], ['EEF'], ax=None)
    # plt.show()
    # print(robot_pose.x, robot_pose.y, robot_pose.z, robot_pose.a, robot_pose.b, robot_pose.c,'<-- xyz, a,b,c')
    T_eef_to_cam = np.array([[1.0,0.,0.,0.],[0., 1.,0.,0],[0.,0.,1., 0],[0.,0.,0.,1.]])
    # plt.show()

    # exit()

    for ctr, T_base_to_goal in enumerate(T_mats):
        # pose_goal_robot_frame = t_mat[0:3,-1].reshape(3,)
        # axis_angles = cv2.Rodrigues(t_mat[0:3,0:3])[0].reshape(-1)
        # goal_in_world_posevec = [pose_goal_robot_frame[0],pose_goal_robot_frame[1],pose_goal_robot_frame[2], \
        #     axis_angles[0], axis_angles[1], axis_angles[2]]
        
        # T_base_to_goal = poseVecToTmat(goal_in_world_posevec)
        T_base_eef_at_goal =  T_base_to_goal  @ rotate_pi_by_2_z_axis_4x4 #invPoseSE3(T_eef_to_cam  @ rotate_pi_by_2_z_axis_4x4)
        _dest = getRvecTvecFromSE3(T_base_eef_at_goal)
        # T_base_to_eef = getEEFPose(robot)

        # _dest_ori, _ = getRvecTvecFromSE3(invPoseSE3(T_base_to_eef) @ T_base_eef_at_goal)


        # print(_dest)
        # exit()
        # goToPose(robot, _dest[1], _dest[0] )
        print(ctr,'<-- step')
        # input("Arm Paused. Press Enter to continue...")
        # goToPose(robot, _dest[1], [-np.pi/4, -np.pi/8 , 0.])
        # goToPose(robot, _dest_pos, _dest_ori)
        # ax = plotFrames([poseVecToTmat([_dest[1][0], _dest[1][1], _dest[1][2], -np.pi/4, -np.pi/8 , 0.])], [str(ctr)], ax=ax)
        ax = plotFrames([poseVecToTmat([_dest[1][0], _dest[1][1], _dest[1][2], _dest[0][0], _dest[0][1], _dest[0][2]])], [str(ctr)], ax=ax)
        # ax = plotFrames([poseVecToTmat([_dest[1][0], _dest[1][1], _dest[1][2], _dest[0][0], _dest[0][1], _dest[0][2]])], [str(ctr)], ax=ax)
        # xarm.sleep(1)
        # time.sleep(cam.ExposureTime/100000 + 0.05) 
        # captureImages(board_ring, ring_pin_IDs, ring_frame_names)
        # posevec_base_tcp = xarm.getBaseToEefPose()
    plt.show()

if __name__ == '__main__':
    # init 
    robot = Robot("172.16.0.2", realtime_config=RealtimeConfig.Ignore)
    robot.set_default_behavior()
    robot.recover_from_errors()
    # robot.set_dynamic_rel(0.15)

    robot.velocity_rel = 2.5
    robot.acceleration_rel = 0.2
    robot.jerk_rel = 0.01

    # known home position
    home_joint_space = [0.0, 0.0, -0.2, -2.642836226488041, -0.0, 3.039206359375316, 0]

    # send to home
    # print("Sending robot to home")
    robot.move(JointMotion(home_joint_space))
    # print("Robot set to home")
    # exit()

    # goToPose(robot, [0.4,0.0,0.3], [0.,0., 0.])
    # goToPose(robot, [0.5,0.0,0.3], [0.,0., 0.])
    # goToPose(robot, [0.4,0.0,0.3], [0.,0., 0.])
    # goToPose(robot, [0.4,0.3,0.3], [0.,0., 0.])

    # goToPose(robot, [0.6,0.0,0.3], [-np.pi/4 ,0., 0.])
    # goToPose(robot, [0.4,0.0,0.3], [0., np.pi/8, 0.])
    # goToPose(robot, [0.4,0.0,0.3], [math.pi/4,0., 0.])
    # goToPose(robot, [0.4,0.0,0.3], [0.,0.,math.pi/4])
    # goToPose(robot, [0.5,0.3,0.3], [0.,math.pi/4, 0.])
    # goToPose(robot, [0.6,0.3,0.3], [0.,math.pi/4, 0.])
    # goToPose(robot, [0.5,0.4,0.3], [0.,math.pi/4, 0.])
    # goToPose(robot, [0.4,0.0,0.3], [math.pi/4,0., 0.])
    # goToPose(robot, [0.5,0.4,0.3], [0.,math.pi/4, math.pi/4])
    # goToPose(robot, [0.45,0.4,0.4], [0.,0., math.pi/4])
    # goToPose(robot, [0.45,0.4,0.4], [0.,0., 0])
    # robot.move(JointMotion(home_joint_space))

    moveCameras([0.875, 0., 0.25,])






