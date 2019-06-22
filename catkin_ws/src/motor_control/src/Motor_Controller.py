#!/usr/bin/env python
#import RPi.GPIO as GPIO
import numpy as np
import rospy
from aquadrone_msgs.msg import MotorControls

pwmData = np.genfromtxt("Motor PWM Width - Thrust Relation.csv", delimiter=",")
#microseconds
pwmSignals = pwmData[:, 0]
#pounds
motorThrusts = pwmData[:, 1]

#in order of placement on prototyping board, starting from the imu
motorPins = [17, 27, 22, 5, 6, 13]
#GPIO.setmode(GPIO.BCM)

pwm = [None, None, None, None, None, None]
#offset to correct systematic error
pwmOffset = 0.3754

for i in range(0, 6):
	pass
	#GPIO.setup(motorPins[i], GPIO.OUT)
	#pwm[i] = GPIO.PWM(motorPins[i], 50)
	#pwm[i].start(7.5 - pwmOffset)


def applyThrusts(motorControls):
	for i in range(0, 6):
		thrustPWM = np.interp(motorControls.motorThrusts[i], motorThrusts, pwmSignals)
		#pwm[i].ChangeDutyCycle(thrustPWM / 200.0 - pwmOffset)
		print(thrustPWM)

#wait for motors to initialize before accepting requests
rospy.sleep(7.0)

rospy.init_node("thruster_hardware_interface")
rospy.Subscriber("motor_command", MotorControls, applyThrusts)
rospy.spin()
	