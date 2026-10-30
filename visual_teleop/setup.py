from setuptools import find_packages, setup

package_name = 'visual_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/perception.launch.py',
            'launch/sim_turtlebot.launch.py',
            'launch/full_system.launch.py',
        ]),
        ('share/' + package_name + '/config', ['config/params.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kunal Dabbe',
    maintainer_email='kunaldabbe2003@gmail.com',
    description='Visual teleoperation package using webcam for target tracking and TurtleBot3 control',
    license='Apache-2.0',
    url='https://github.com/KuNaL8103/VisionBot',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'perception_node = visual_teleop.perception_node:main',
            'controller_node = visual_teleop.controller_node:main',
        ],
    },
)