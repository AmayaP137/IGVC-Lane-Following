from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'lane_follower'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='CNN lane following for ROS 2 Jazzy on Raspberry Pi 5',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node    = lane_follower.camera_node:main',
            'detector_node  = lane_follower.detector_node:main',
            'controller_node = lane_follower.controller_node:main',
        ],
    },
)
