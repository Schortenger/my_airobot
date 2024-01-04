from yacs.config import CfgNode as CN

_C = CN()

# prefix of the class name of the ARM
# if it's for pybullet simulation, the name will
# be augemented to be '<Prefix>Pybullet'
# if it's for the real robot, the name will be
# augmented to be '<Prefix>Real'
_C.CLASS = 'Franka'

# https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/max-joint-torques-17260/
_C.MAX_TORQUES = [87, 87, 87, 87, 12, 12, 12, 100, 100]
_C.JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
# base frame for the arm
_C.ROBOT_BASE_FRAME = 'panda_link0'
# end-effector frame of the arm
_C.ROBOT_EE_FRAME = 'panda_grasptarget'
_C.ROBOT_EE_FRAME_JOINT = 'panda_grasptarget_hand'

# inverse kinematics position tolerance (m)
_C.IK_POSITION_TOLERANCE = 0.01
# inverse kinematics orientation tolerance (rad)
_C.IK_ORIENTATION_TOLERANCE = 0.05
# _C.HOME_POSITION = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0, 0]
# _C.HOME_POSITION = [-1.6, 0.64, -0.22, -2.21, 0.47, 2.76, -1.36, 0, 0]
_C.HOME_POSITION = [-1.887906702852056, 0.08481914207986918, -0.25098564412533175, -2.092232178931657, 0.05247358344411879, 2.12395499469104, -1.323048507159928, 3.4127012633611287e-06, -6.393174603047669e-07]
# _C.HOME_POSITION = [-0.19, 0.08, 0.23, -2.43, 0.03, 2.52, 0.86, 0, 0]
_C.MAX_JOINT_ERROR = 0.01
_C.MAX_JOINT_VEL_ERROR = 0.05
_C.MAX_EE_POS_ERROR = 0.01
# real part of the quaternion difference should be
# greater than 1-error
_C.MAX_EE_ORI_ERROR = 0.02
_C.TIMEOUT_LIMIT = 10

# reset position for the robot in pybullet
_C.PYBULLET_RESET_POS = [0, 0, 1]
# reset orientation (euler angles) for the robot in pybullet
_C.PYBULLET_RESET_ORI = [0, 0, 0]
_C.PYBULLET_IK_DAMPING = 0.0005


def get_franka_arm_cfg():
    return _C.clone()
