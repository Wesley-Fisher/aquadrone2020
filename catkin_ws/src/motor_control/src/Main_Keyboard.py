#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import Wrench

print("Starting")
#wait for motors to set up
rospy.sleep(7)

rospy.init_node("main")
publisher = rospy.Publisher("movement_command", Wrench, queue_size=0)

while True:
	x = 0
	z = 0
	yaw = 0

	if rospy.is_shutdown():
		exit()
	
	control = raw_input("Command:")
	if control == "w":
		x = 1
	elif control == "s":
		x = -1
	elif control == "a":
		yaw = 1
	elif control == "d":
		yaw = -1
	elif control == "i":
		z = 1
	elif control == "k":
		z = -1
	
	command = Wrench()
	command.force.x = x
	command.force.z = z
	command.torque.z = yaw 
	publisher.publish(command)