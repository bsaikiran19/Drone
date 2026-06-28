from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'opencv_drone_vision'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'models'),
            glob('models/**/*', recursive=True)),
        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='drone_dev',
    maintainer_email='drone@example.com',
    description='Vision-Based Dynamic Obstacle Avoidance for UAV using OpenCV',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'opencv_navigation = opencv_drone_vision.opencv_navigation:main',
            'moving_obstacle = opencv_drone_vision.moving_obstacle:main',
        ],
    },
)
