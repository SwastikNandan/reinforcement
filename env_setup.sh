# copy model files
cp -r ~/catkin_ws/src/reinforcement/helpers/models/* ~/.gazebo/models

# copy basket mesh files
cp -r ~/catkin_ws/src/reinforcement/helpers/turtlebot/basket ~/catkin_ws/src/turtlebot3/turtlebot3_description/meshes/

# copy xacro file
cp -r ~/catkin_ws/src/reinforcement/helpers/turtlebot/turtlebot3_waffle.urdf.xacro ~/catkin_ws/src/turtlebot3/turtlebot3_description/urdf/

# catkin make
cd ~/catkin_ws && catkin_make

# source latest
source ~/catkin_ws/devel/setup.bash