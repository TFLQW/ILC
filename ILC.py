import numpy as np
from matplotlib import pyplot as plt
import seaborn
import time

class ILC(object):
    """
    L: init matrix of gain
    Q: init mattix of filter
    iter_num: default iterative cycles
    time_step: default the long time of each cycle
    degree: the degree of the control variable
    """
    def __init__(self, L, Q, timestep=100, degree=6):

        # if not isinstance(L, list):
        #     raise TypeError("Expecting 'L' to be an instance of 'List', but got instead: " "{}".format(type(L)))
        self.L = L

        # if not isinstance(Q, list):
        #     raise TypeError("Expecting 'Q' to be an instance of 'List', but got instead: " "{}".format(type(Q)))
        self.Q = Q

        self.timestep = timestep
        self.degree = degree
        self.u = np.zeros((self.timestep, self.degree))
        # self.record_flag = False
    def Read_Record(self, timestep):
        return self.u[timestep, :]

    def ILC_Record(self, input, timestep):
        if not isinstance(input, list):
            raise TypeError("Expecting 'input' to be an instance of 'list', but got instead: "
                            "{}".format(type(input)))
        if len(input) != self.degree:
            raise Exception("The control number is different from you have set!")
        if timestep > self.timestep:
            raise Exception("Out of the setting of time")

        self.u[timestep, :] = input

        # if time_step == self.time_step:
            # self.record_flag = True
    def ILC_Control(self, feedback_error, timestep):
        if not isinstance(feedback_error, list):
            raise TypeError("Expecting 'feedback' to be an instance of 'list', but got instead: " 
                            "{}".format(type(feedback_error)))

        u = np.dot(self.Q, (self.u[timestep, :] + np.dot(self.L, feedback_error)))

        return u

if __name__=="__main__":
    LT = 3001
    N = 5
    time = np.linspace(0, 3, LT)
    T = np.zeros((LT, 2))
    e = np.zeros((LT, 4))
    q1d = np.sin(3 * time)
    q2d = np.cos(3 * time)
    dq1d = 3 * np.cos(3 * time)
    dq2d = -3 * np.sin(3 * time)
    Q = 3 * np.identity(2)
    L = 5 * np.identity(2)
    ILC_handle = ILC(Q=Q, L=L, timestep=LT, degree=2)
    m1 = 10
    m2 = 5
    I1 = 1
    I2 = 0.5
    r1 = 0.5
    r2 = 0.25
    i1 = 0.83 + m1 * np.square(r1) + m2 * np.square(I1)
    i2 = 0.3 + m2 * np.square(r2)
    g = 9.8

    color_map = ['light blue grey', 'powder blue',  'baby blue', 'sky blue', 'dark sky blue', 'clear blue', 'bright blue', 'medium blue', 'windows blue', 'mid blue']

    plt.ion()
    fig1, ax1 = plt.subplots(nrows=2, ncols=1)

    e_ex = np.zeros((N+1, 2))
    de_ex = np.zeros((N+1, 2))
    for i in range(N):
        q1 = 0
        q2 = 1
        dq1 = 3
        dq2 = 0
        q = np.zeros((LT, 2))  # store the every cycle data: q
        dq = np.zeros((LT, 2))  # store every cycle data: dq
        e = np.zeros((LT, 2))
        de = np.zeros((LT, 2))
        q[0, :] = [q1, q2]
        dq[0, :] = [dq1, dq2]
        # plt.ion()
        for t in range(LT):
            dts = 0.001
            dt = t * dts
            # a = 1000 * np.random.randn(1)[0]
            a = 1.0

            d1 = a * 0.3 * np.sin(3*dt)  # damping term
            d2 = a * 0.1 * (1-np.exp(-dt))  # damping term

            e1 = q1d[t] - q1
            e2 = q2d[t] - q2
            de1 = dq1d[t] - dq1
            de2 = dq2d[t] - dq2
            e[t, :] = [e1, e2]
            de[t, :] = [de1, de2]

            Fai = np.identity(2)
            Kd0 = np.array([[210, 0], [0, 210]])
            beta = 2
            # if i == 0:
            #     beta = 1
            # else:
            #     beta = 2
            sys = [beta*210*(e1+de1), beta*110*(e2+de2)]
            tol1 = ILC_handle.Read_Record(t)[0]
            tol2 = ILC_handle.Read_Record(t)[1]

            tol1 = tol1 + sys[0]
            tol2 = tol2 + sys[1]

            ILC_handle.ILC_Record([tol1, tol2], t)

            D = np.array([[i1 + i2 + 2 * m2 * r2 * I1 * np.cos(q2), i2 + m2 * r2 * I1 * np.cos(q2)], [i2 + m2 * r2 * I1 * np.cos(q2), i2]])
            C = np.array([[-m2 * r2 * I1 * dq2 * np.sin(q2), -m2 * r2 * I1 * (dq1 + dq2) * np.sin(q2)], [m2 * r2 * I1 * dq1 * np.sin(q2), 0]])
            G = np.array([[(m1 * r1 + m2 * I1) * g * np.cos(q1) + m2 * r2 * g * np.cos(q1+q2)], [m2 * r2 * g * np.cos(q1 + q2)]])
            D2 = np.linalg.inv(D)
            Ta = np.vstack((d1, d2))
            A = np.dot(-D2, C)
            Z = np.dot(-D2, G)
            ddq1 = (A[0, 0] * dq1 + A[0, 1] * dq2 + Z[0] + D2[0, 0] * (-Ta[0, 0] + tol1) + D2[0, 1]*(-Ta[1, 0] + tol2))[0]
            ddq2 = (A[1, 0] * dq1 + A[1, 1] * dq2 + Z[1] + D2[1, 0] * (-Ta[0, 0] + tol1) + D2[1, 1]*(-Ta[1, 0] + tol2))[0]
            dq1 = dq1 + ddq1 * dts
            dq2 = dq2 + ddq2 * dts
            q1 = q1 + dq1 * dts
            q2 = q2 + dq2 * dts
            q[t, :] = [q1, q2]
            dq[t, :] = [dq1, dq2]
            pass
        # q_ex.append(q)
        # dq_ex.append(dq)
        # ax = fig.add_subplot(2, 1, 1)
        e_ex[i, :] = np.mean(e, axis=0)
        ax1[0].plot(time.reshape(time.shape[0], 1), q[:, 0], color=seaborn.xkcd_rgb[color_map[i]])
        ax1[0].plot(time.reshape(time.shape[0], 1), q1d, color='k', linestyle='--')
        # ax = fig.add_subplot(2, 1, 2)
        ax1[1].plot(time.reshape(time.shape[0], 1), q[:, 1], color=seaborn.xkcd_rgb[color_map[i]])
        ax1[1].plot(time.reshape(time.shape[0], 1), q2d, color='k', linestyle='--')
        fig1.canvas.flush_events()

        plt.pause(0.1)
    fig2, ax2 = plt.subplots()

    iter = np.linspace(0, N, N+1)
    it = iter.reshape(iter.shape[0], 1)
    ax2.plot(iter.reshape(iter.shape[0], 1), e_ex[:, 0], color='r')
    ax2.plot(iter.reshape(iter.shape[0], 1), e_ex[:, 1], color='b')
    ax2.scatter(iter.reshape(iter.shape[0], 1), e_ex[:, 0], c='g', marker='*')
    ax2.scatter(iter.reshape(iter.shape[0], 1), e_ex[:, 1], c='g', marker='*')
    plt.ioff()
    plt.show()
