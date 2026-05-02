import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/amaya28/ros2_ws/src/lanenet_ros/install/lanenet_ros'
