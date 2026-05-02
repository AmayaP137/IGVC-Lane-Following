from setuptools import setup
import os
from glob import glob

package_name = 'lanenet_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # This line includes all launch files (if you create them later)
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # This line ensures your weights are copied to the install directory
        (os.path.join('share', package_name, 'weights'), glob('weights/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@todo.todo',
    description='ROS 2 wrapper for LaneNet',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detector = lanenet_ros.lane_detector_node:main',
        ],
    },
)