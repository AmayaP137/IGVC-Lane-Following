from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'IGVC_lane_follow'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # launch files
        ('share/' + package_name + '/launch',
            glob('launch/*.py')),

        # config
        ('share/' + package_name + '/config',
            glob('config/*.yaml')),

       
      # onnx model
        ('share/' + package_name + '/models',
            glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='CNN lane detection and following for ROS 2 Jazzy',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            
            'detector_node   = IGVC_lane_follow.detector_node:main',
            'controller_node = IGVC_lane_follow.controller_node:main',
        ],
    },
)
