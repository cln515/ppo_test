import os
import pybullet as p2
import pybullet_data
from pybullet_utils import bullet_client as bc
import numpy as np
import math

class cartpole:
    def __init__(self, renders=True):
        self._physics_client_id = -1
        self._renders= renders
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 0.4
        return

    def reset(self):
        if self._physics_client_id < 0:
            if self._renders:
                self._p = bc.BulletClient(connection_mode=p2.GUI)
            else:
                self._p = bc.BulletClient()
            self._physics_client_id = self._p._client
        
            p = self._p
            p.resetSimulation()
            self.cartpole = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cartpole.urdf"),
                                        [0, 0, 0])
            p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0)
            p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0)
            p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0)
            self.timeStep = 0.02
            p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self.timeStep)
            p.setRealTimeSimulation(0)
        p = self._p
        #randstate = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        randstate = np.random.random_sample(4)
        randstate = randstate * 0.01 - 0.005 
        p.resetJointState(self.cartpole, 1, randstate[0], randstate[1])
        p.resetJointState(self.cartpole, 0, randstate[2], randstate[3])
        #print("randstate=",randstate)
        self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]
        #print("self.state=", self.state)
        return np.array(self.state)

    def step(self, action):
        p=self._p
        force = action[0]
        p.setJointMotorControl2(self.cartpole,0,p.TORQUE_CONTROL, force=force)
        p.stepSimulation()

        self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2]

        theta, theta_dot, x, x_dot = self.state

        done =  x < -self.x_threshold \
                    or x > self.x_threshold \
                    or theta < -self.theta_threshold_radians \
                    or theta > self.theta_threshold_radians
        done = bool(done)
        reward = 1.0

        return np.array(self.state), reward, done