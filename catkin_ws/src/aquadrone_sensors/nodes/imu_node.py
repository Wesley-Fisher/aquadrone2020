#!/usr/bin/env python
import rospy
from aquadrone_msgs.msg import AquadroneIMU
from sensor_msgs.msg import Imu
from Adafruit_BNO005 import BNO005

bno = BNO005.BNO055()

rospy.init_node('aquadrone_IMU')
pubUW = rospy.Publisher('/aquadrone_v2/out/SensorUW', AquadroneIMU)
pubIMU = rospy.Publisher('/aquadrone_v2/out/imu', Imu)
rate = rospy.Rate(10)

while not rospy.is_shutdown():
  msgUW = AquadroneIMU()
  euler1,euler2,euler3 = bno.read_euler()
  msgUW.euler.x = euler1
  msgUW.euler.y = euler2
  msgUW.euler.z = euler3
  gyro1,gyro2,gyro3 = bno.read_gyroscope()
  msgUW.gyroscope.x = gyro1
  msgUW.gyroscope.y = gyro2
  msgUW.gyroscope.z = gyro3
  accel1, accel2, accel3 = bno.read_accelerometer()
  msgUW.accelerometer.x = accel1
  msgUW.accelerometer.y = accel2
  msgUW.accelerometer.z = accel3
  linAccel1, linAccel2, linAccel3 = bno.read_linear_acceleration()
  msgUW.linear_acceleration.x = linAccel1
  msgUW.linear_acceleration.y = linAccel2
  msgUW.linear_acceleration.z = linAccel3
  quat1, quat2, quat3, quat4 = bno.read_quaternion()
  msgUW.quaternion.x = quat1
  msgUW.quaternion.y = quat2
  msgUW.quaternion.z = quat3
  msgUW.quaternion.w = quat4
  msgIMU = Imu()
  msgIMU.orientation.x = quat1
  msgIMU.orientation.y = quat2
  msgIMU.orientation.z = quat3
  msgIMU.orientation.w = quat4
  msgIMU.angular_velocity.x = gyro1
  msgIMU.angular_velocity.y = gyro2
  msgIMU.angular_velocity.x = gyro3
  msgIMU.linear_acceleration.x = linAccel1
  msgIMU.linear_acceleration.y = linAccel2
  msgIMU.linear_acceleration.z = linAccel3
  pubUW.publish(msgUW)
  pubIMU.publish(msgIMU)
  rate.sleep()