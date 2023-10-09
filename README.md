#Perception Assesment

##Dependencies:
ROS Version: Melodic
Opencv

##
1.I have used a simple differential drive mobile robot with LIDAR and  Camera.
2.Created a gazebo world with few pedestrians and passed it in parm duting the launch.
3.Included the corresponding  gazebo plugins to get thr required data.
4.Used opencv to visualize the camera data.(there was a version mismatch in cv_bridge, so I had to write the function in code.)
5.Trained a yolov5 object detection model using robobflow for detection of a person standing in gazebo.
6.Loaded the model in opvencv dnn and used it to detect pedestrians from the live feed.
7. Added rqt steering to control the robot.

##
To run,
```
roslaunch robot_description model_gazebo.launch
```
Add terminal
```
rosrun robot_description camera_read.py
```
The robot can be controlled using the rqt steering.