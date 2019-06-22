#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Wrench
from aquadrone_msgs.msg import MotorControls

#assumptions:
#	see diagrams for coordinate system and motor numbering
#	positive thrust aims towards the negative z for motors 0, 1, 4, and 5
#	positive thrust aims directly in the positive x direction for motors 2 and 3
#	positive pitch, roll, and yaw are determined via the corresponding axis using the right hand rule

#inches
xLength = 36
yLength = 12

#all of these are normalized to produce a net force of 1 pound
relativeXThrusts = np.array([0, 0, 1, 1, 0, 0]) / 2
relativeYThrusts = np.array([0, 0, 0, 0, 0, 0]) #not possible
relativeZThrusts = np.array([-1, -1, 0, 0, -1, -1]) / 4

#all of these are normalized to produce a net torque of 1 pound-inch
relativeRollThrusts = np.array([-1, 1, 0, 0, -1, 1]) / (yLength * 2)
relativePitchThrusts = np.array([1, 1, 0, 0, -1, -1]) / (xLength * 2)
relativeYawThrusts = np.array([0, 0, -1, 1, 0, 0]) / yLength

thrustMatrix = np.column_stack((relativeXThrusts, relativeYThrusts, relativeZThrusts, 
			relativeRollThrusts, relativePitchThrusts, relativeYawThrusts))

rospy.init_node("movement_controller")
rospy.Subscriber("movement_command", Wrench, processThrusts)
publisher = rospy.Publisher("motor_command", MotorControls, queue_size=0)

#all inputs in pounds or pound-inches respectively
#note that with this design, the yThrust is ignored as it is not possible
def processThrusts(wrench):
	thrusts = np.array([wrench.force.x, wrench.force.y, wrench.force.z, 
		wrench.torque.x, wrench.torque.y, wrench.torque.z])

	command = MotorControls()
	command.motorThrusts = thrustMatrix.dot(vector)
	publisher.publish(command)