import mujoco as mj
import math
import numpy as np
from scipy.linalg import logm

class NRIK:

    def __init__(self, model, data, site):
        # Mujoco
        self.model = model
        self.data = data
        self.site = site
        self.numJoints = model.nv
        # Jacobian Setup
        self.jacp = np.zeros((3,self.numJoints))
        self.jacr = np.zeros((3,self.numJoints))

    # Inverse Kinematics for Position and Pose
    def solveIK_full_arm(self, endPos, endRot):

       
    # Inverse Kinematics for Position Only
    def solveIK_3dof(self, endPos):

            
    def getBodyJacobian(self):


    def adjointMatrix(self):