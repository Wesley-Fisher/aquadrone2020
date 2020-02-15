#!/usr/bin/env python

import rospy
import math
import time
#import numpy as np
import scipy.linalg
import random

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad, jacobian, elementwise_grad

from geometry_msgs.msg import Point, Vector3, Quaternion
from sensor_msgs.msg import Imu, FluidPressure

from thruster_control.configurations.v28_configuration import V28Configuration
from aquadrone_msgs.msg import SubState, MotorControls

import matplotlib.pyplot as plt


class IDx:
    # Position
    Rx = 0
    Ry = 1
    Rth = 2

    Mx = 3
    My = 4
    Mth = 5
    
    L1x = 6
    L1y = 7

    L2x = 8
    L2y = 9

    L3x = 10
    L3y = 11

    L4x = 12
    L4y = 13


class LandmarkSensor(object):
    # https://en.wikipedia.org/wiki/Extended_Kalman_filter

    #
    # Functions Common for Each Sensor; No need to re-implement
    #

    def __init__(self, i, x0, y0, xm, ym, use_map):
        self.last_time = time.time()
        self.calc_H = jacobian(self.state_to_measurement_h)

        self.z_dx = 0
        self.z_dy = 0

        self.use_map = use_map

        self.ix = IDx.L1x + 2*i
        self.iy = self.ix + 1

        # True Locations
        self.x0 = x0
        self.y0 = y0

        # Starting Location for x-vector
        self.xm0 = xm
        self.ym0 = ym


    def is_valid(self):
        return time.time() - self.last_time < self.get_timeout_sec()

    def get_H(self, x, u):
        # Jacobian of measurement wrt state (as calculated by get_measurement_z)
        H = self.calc_H(x, u)
        H = np.reshape(H, (self.get_p(), x.shape[0]))
        return H


    def get_timeout_sec(self):
        return 0.1

    def get_p(self):
        return 2

    def set_measurement(self, xr, yr, r):
        self.last_time = time.time()
        self.z_dx = self.x0 - xr + (random.random()-0.5)*2*r
        self.z_dy = self.y0 - yr + (random.random()-0.5)*2*r

    def get_measurement_z(self):
        # Actual measurement from the sensor
        vec = np.zeros((2,1))
        vec[0] = self.z_dx
        vec[1] = self.z_dy
        return vec

    def get_R(self):
        var = np.zeros((2,2))
        var[0][0] = 0.5
        var[1][1] = 0.5
        return var

    def state_to_measurement_h(self, x, u):
        # Calculate expected sensor readings from current state and inputs
        # Needs to be autograd-able
        if self.use_map:
            th = x[IDx.Mth]
            x_land = x[IDx.Mx] + np.cos(th)*x[self.ix] - np.sin(th)*x[self.iy]
            y_land = x[IDx.My] + np.sin(th)*x[self.ix] + np.cos(th)*x[self.iy]
            return np.array([[x_land - x[IDx.Rx]],
                            [y_land - x[IDx.Ry]]])

        else:
            return np.array([[x[self.ix] - x[IDx.Rx]],
                             [x[self.iy] - x[IDx.Ry]]])



class EKF:
    def __init__(self, landmarks, dt):
        # https://en.wikipedia.org/wiki/Extended_Kalman_filter

        ''' Model Description
        x = state
        u = inputs (thruster forces, etc)
        z = ouputs (measurements)
        P = unvertainty/variance matrix of state x

        Dynamics: x[k+1] = f(x[k], u[k])
        Outputs:    z[k] = h(x[k], u[k])

        Linear Form:
          x[k+1] = A*x[k] + B*u[k]
            z[k] = C*x[k] + B*u[k]
        '''

        ''' Process Description

        Each loop will do a prediction step based on the motor thrusts,
        gravity, bouyancy, drag, and other forces to update the expected
        state and its variance.
        
        Then a measurement step, where it uses input from the pressure sensor
        and gyro (and sensors added in hte future) to refine its expected state
        and variance.
        
        Then there is a step where the new expected state/variance is converted
        to a SubState message which is then published.

        '''

        self.dt = dt

        self.n = 16 # Number of state elements
        self.m = 1 # Number of inputs

        self.x = np.zeros((self.n, 1))
        
        self.landmarks = landmarks

        self.u = np.zeros((self.m,1))

        #self.B = np.array(config.get_thrusts_to_wrench_matrix())

        self.P = np.eye(self.n)
        self.P[IDx.Mx][IDx.Mx] = 20
        self.P[IDx.My][IDx.My] = 20
        self.P[IDx.Mth][IDx.Mth] = 20
        self.Q = np.eye(self.n) * 0.00001 # Uncertanty in dynamics model

        
        # Potential Future Sensors:
        # - ZED mini localization
        # - ZED mini IMU
        # - Position info from detecting objects

        self.rate = 20
        #self.rate_ctrl = rospy.Rate(self.rate)

        self.calc_F = jacobian(self.f)

        self.sub_state_msg = SubState()
        self.last_prediction_t = self.get_t()

    def motor_cb(self, msg):
        self.u = np.array(msg.motorThrusts)

    def get_t(self):
        return time.time()

    def prediction(self):

        # Get jacobian of function f
        F = self.calc_F(self.x, self.u)
        F = np.reshape(F, (self.n,self.n))
        Fx = F[0:self.n, 0:self.n]
        #print("Fx")
        #print(Fx)

        # Update x and uncertainty P
        self.x = self.f(self.x, self.u)
        inter = np.dot(Fx, self.P)
        self.P = np.dot(  inter,  np.transpose(Fx)  ) + self.Q

        

    def update(self):
        # Update state based on sensor measurements
        z = np.zeros((0,0)) # measurements
        h = np.zeros((0,0)) # predicted measurements

        H = np.zeros((0,0)) # Jacobian of function h
        R = np.zeros((0,0)) # Uncertainty matrix of measurement

        def add_block_diag(H, newH):
            if H.shape[0] == 0:
                #ret = np.diag(newH)
                #ret.shape = (newH.shape[0],newH.shape[0])
                return newH
            return scipy.linalg.block_diag(H, newH)

        def add_block_vertical(H, newH):
            if H.shape[0] == 0:
                return newH
            return np.vstack([H, newH])

        def read_listener(listener, z, h, H, R):
            if listener.is_valid():
                meas = listener.get_measurement_z()
                z = np.append(z, np.array([meas]))
                pred = listener.state_to_measurement_h(self.x, self.u)
                h = np.append(h, np.array([pred]))

                H = add_block_vertical(H, listener.get_H(self.x, self.u))
                R = add_block_diag(R, listener.get_R())

                return z, h, H, R
            return z, h, H, R

        for listener in self.landmarks:
            try:
                z, h, H, R = read_listener(listener, z, h, H, R)
                
            except TypeError as e:
                print(e)
                return

        if R.shape[0] == 0:
            return

        # Error in measurements vs predicted
        y = z - h
        y.shape = (y.shape[0], 1)

        # Calculate Kalman gain
        Ht = np.transpose(H)
        S = np.dot(np.dot(H, self.P), Ht) + R
        K = np.dot(np.dot(self.P, Ht), np.linalg.inv(S))

        KH = np.dot(K, H)
        I = np.eye(KH.shape[0])

        diff = np.dot(K, y)

        # Update state x and uncertainty P
        self.x = self.x + diff
        self.P = np.dot(I - KH, self.P)


    def f(self, x, u):
        # Calculate next state from current state x and inputs u
        # Must be autograd-able

        robot_state = np.array([x[IDx.Rx],
                                x[IDx.Ry],
                                x[IDx.Rth]])

        Mx = x[IDx.Mx]
        My = x[IDx.My]

        scalar = math.exp(-0.1*self.dt)

        Mx_new = 0.9*x[IDx.Mx]
        My_new = 0.9*x[IDx.My]
        
        map_state = np.array([ Mx_new,
                               My_new,
                               x[IDx.Mth] ])

        newx = np.vstack([robot_state, map_state])
        #print(newx)

        for L in self.landmarks:
            x0 = x[L.ix]
            y0 = x[L.iy]

            dx = x0 + Mx - Mx_new
            dy = y0 + My - My_new

            state = np.array([ dx, dy])

            newx = np.vstack([newx, state])

        return newx



    def output(self):
        var = np.array(np.diagonal(self.P))
        var.shape = (self.n, 1)
        
        #print("X, Var:")
        #print(np.hstack([self.x, var]))

            
            

if __name__ == "__main__":
    
    
    
    dt = 0.01
    
    passes = 2
    T = 10

    rx0 = 0
    ry0 = 0


    dx = 10
    dy = 3

    def make_landmark_pair(i, x0, y0, xm, ym):
        L = LandmarkSensor(i,  x0,  y0,  xm,  ym, True)
        L_nomap = LandmarkSensor(i,  x0,  y0,  xm,  ym, False)
        return L, L_nomap

    L1, L1_nomap = make_landmark_pair(0,  1,  0,  1+dx, 0+dy)
    L2, L2_nomap = make_landmark_pair(1,  0,  1,  0+dx,  1+dy)
    L3, L3_nomap = make_landmark_pair(2, -1,  0, -1+dx,  0+dy)
    L4, L4_nomap = make_landmark_pair(3,  0, -1,  0+dx, -1+dy)
    L5, L5_nomap = make_landmark_pair(4,  1,  1,  1+dx,  1+dy)

    landmarks = [L1, L2, L3, L4, L5]

    landmarks_nomap = [L1_nomap, L2_nomap, L3_nomap, L4_nomap, L5_nomap]


    ekf = EKF(landmarks, dt)
    ekf_nomap = EKF(landmarks_nomap, dt)

    for L in landmarks:
        ekf.x[L.ix] = L.xm0
        ekf.x[L.iy] = L.ym0

        ekf_nomap.x[L.ix] = L.xm0
        ekf_nomap.x[L.iy] = L.ym0

    plot_sizes = [4, 9, 16, 25, 36]
    
    for i in range(0, passes):
        t0 = time.time()
        dT = 0
        while dT < T:
            dT = time.time() - t0
            print(dT)

            phase = 2*math.pi * 0.1 * dT
            radius = 0.25

            rx = radius*np.cos(phase)
            ry = radius*np.sin(phase)

            
            if dT < T/4.0:
                L1.set_measurement(rx, ry, 0.05)
                L1_nomap.set_measurement(rx, ry, 0.5)
                L2.set_measurement(rx, ry, 0.05)
                L2_nomap.set_measurement(rx, ry, 0.5)
            elif dT < 2*T/4.0:
                L2.set_measurement(rx, ry, 0.05)
                L2_nomap.set_measurement(rx, ry, 0.5)
                L3.set_measurement(rx, ry, 0.05)
                L3_nomap.set_measurement(rx, ry, 0.5)
            elif dT < 3*T/4.0:
                L3.set_measurement(rx, ry, 0.05)
                L3_nomap.set_measurement(rx, ry, 0.5)
                L4.set_measurement(rx, ry, 0.05)
                L4_nomap.set_measurement(rx, ry, 0.05)
            elif dT < 4*T/4.0:
                L4.set_measurement(rx, ry, 0.05)
                L4_nomap.set_measurement(rx, ry, 0.05)
            
            ekf.prediction()
            ekf.update()

            ekf_nomap.prediction()
            ekf_nomap.update()

            def plot_col(idx, landmarks, ekf):

                plt.subplot(2,3,1+3*idx)
                x = [float(L.x0) for L in landmarks]
                y = [float(L.y0) for L in landmarks]
                plt.scatter(x, y, plot_sizes, c='b')
                plt.scatter([rx], [ry], c='r')
                plt.title("Ground Truth")



                plt.subplot(2,3,2+3*idx)
                plt.cla()
                x = [float(ekf.x[L.ix]) for L in landmarks]
                y = [float(ekf.x[L.iy]) for L in landmarks]
                plt.scatter(x, y, plot_sizes, c='b')
                plt.scatter(ekf.x[IDx.Rx], ekf.x[IDx.Ry], c='r')
                plt.title("Map Local from x")


                plt.subplot(2,3,3+3*idx)
                plt.cla()
                loc_x = [float(ekf.x[L.ix]) for L in landmarks]
                loc_y = [float(ekf.x[L.iy]) for L in landmarks]

                th = ekf.x[IDx.Mth]
                wx = []
                wy = []
                for x, y in zip(loc_x, loc_y):
                    ox = ekf.x[IDx.Mx] + np.cos(th)*x - np.sin(th)*y
                    oy = ekf.x[IDx.My] + np.sin(th)*x + np.cos(th)*y

                    wx.append(ox)
                    wy.append(oy)

                plt.scatter(wx, wy, plot_sizes, c='b')
                plt.scatter(ekf.x[IDx.Rx], ekf.x[IDx.Ry], c='r')
                plt.title("Est World from x [Transformed Map->World]")

            plot_col(0, landmarks, ekf)
            plot_col(1, landmarks_nomap, ekf_nomap)

            plt.pause(0.001)

            time.sleep(dt)


    L1.set_measurement(0, 0, 0)
    print(L1.z_dx)
    print(L1.z_dy)

    a = ekf.x
    print(a.shape)
    b = np.diag(ekf.P)
    b = b.reshape(b.shape[0],1)
    print(b.shape)
    c = np.hstack([a, b])
    print(c)
        
    plt.show()