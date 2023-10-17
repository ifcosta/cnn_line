# CNN Line Package Readme
Overview
The CNN Line package is designed for ROS1 (Noetic). This package is created to facilitate row following and control of a mobile robot in an agricultural environment.

## Simulation Execution

### rosrun cnn_line new_world.py
This script creates a new random field with each execution, and the models used are located inside the models folder in the cnn_line package. The script is responsible for managing all the necessary paths.

### roslaunch cnn_line simulation.launch
This launch file initiates the simulation environment and places the robot in the field.

### cnn_line joycmd.launch joy_dev:=js0
Starts the ROS joy_node package to interface with the joystick hardware. The argument joy_dev points to the correct joystick to be used (default js0).

### roslaunch cnn_line joy_node.launch
Initiates the main control node.

### roslaunch cnn_line line_control.launch
It is responsible for launching the line control system, receiving a line from line_detection to guide the robot.

### roslaunch cnn_line line_detection.launch
It initiates the line detection system; line following still needs to be enabled.

## Robot Execution
To test with a real-world robot, configure the correct topics for the `joy_node`, `line_detection`, and `line_control`. Refer to the launch files for parameter and topic instructions.
The main areas of concern are the joystick device path, the image topic, and the velocity topic. Verify them in the `joycmd.launch`, `line_control.launch`, and `line_detection.launch`.

The `joycmd.launch` file might cause incompatibilities if the target robot is already using a joystick. In such cases, omit it and point the `joy` topic to the joystick topic provided by the target robot in the `joy_node.launch`.

Be cautious about potential button conflicts.

## Button Layout
Start and Select Buttons (Start + Select): Toggles autonomous mode when both buttons are pressed together. The status of the autonomous mode is logged.

A Button: Initiates the home steering process when pressed. (Note: This feature is deactivated. Thorvald platform only)
L Button: Decreases the speed gain by 0.1 when pressed. Updates the ROS parameter /linear_velocity with the new speed gain.
R Button: Increases the speed gain by 0.1 when pressed. Updates the ROS parameter /linear_velocity with the new speed gain.
(A + B): Toggles between control types ('cascade' and 'row_control') when both buttons are pressed simultaneously. It updates the parameter server with the new control type.

Left Analog Stick: Control the robot, forward/backward, and turn right/left.

Refer to the code for further informations

## Getting Started
### Preparing the Data
Training Folder (inside cnn_line): The training folder contains scripts and tools to create the dataset. The notebook for training the model is also inside it.

To use place the dataset in the following structure:
|cnn_line_train.ipynb
|label.py

|*dataset_folder*
||*test*
|||*images*

||*train*
|||*images*

||*val*
|||*images*

run label.py to initiate the labeling script

### Labeling
Two windows will open; the smaller window is used to draw the reference line, and the bigger window is for building the mask. An example can be seen in the root of the repository.

Visualization Toggle:
- Base: Original Image
- Proc: Image with all processing applied
- RGB: Color-coded mask
- Cat: Categoric-coded mask

Bot Camera toggle: This option makes the red mask start at the beginning of the drawn line in the drawing window.

- Line geometry slider: Controls the size of the red area around the line.
- EX Threshold: Threshold sensitivity for the Excess of Green - Excess of Red algorithm used.
- EX Green: Excess of green parameter gain.
- EX Red: Excess of red parameter gain.
- CLAHE: Clahe image preprocessing parameters for contrast boost.
- HSV: HSV image gains for simple color balance and saturation improvements.
- BLUR: Blur applied to simplify the mask; Erode and Dilate applied in this order to the green mask only.
- FILTER: Keeps only the X largest mask elements to filter smaller and undesired areas.
- SKY: Enables the sky mask (Bot Camera toggle needed).

The following keys control the program (While the drawing window is in focus):
- S key skips the image
- N key goes to the next image
- Q key quits the program



