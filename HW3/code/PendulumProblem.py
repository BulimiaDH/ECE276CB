"""
==================================
Inverted pendulum animation class
==================================

Adapted from the double pendulum problem animation.
https://matplotlib.org/examples/animation/double_pendulum_animated.html
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
from scipy import interpolate
import pickle
import time
import os


class EnvAnimate:
    '''
    Initialize Inverted Pendulum
    '''

    def __init__(self, dt=0.05, n1=200, n2=40, nu=40):
        # Change this to match your discretization
        self.dt = dt
        self.t = np.arange(0.0, 15.0, self.dt)
        self.n1 = n1
        self.n2 = n2
        self.nu = nu
        self.thetamesh = 2 * np.pi / self.n1
        self.umax = math.pi * self.nu / (self.n1 * self.dt)
        self.umesh = 2 * self.umax / self.nu
        print(self.umax, self.umesh)
        self.vmax = math.pi * self.n2 / (self.n1 * self.dt)
        self.vmesh = 2 * self.vmax / self.n2
        print(self.vmax, self.vmesh)

        # Random trajectory for example
        self.theta = np.linspace(-np.pi, np.pi, self.t.shape[0])
        self.x1 = np.sin(self.theta)
        self.y1 = np.cos(self.theta)
        self.u = np.zeros(self.t.shape[0])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])

        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.8, '', transform=self.ax.transAxes)

    '''
    Provide new rollout theta values to reanimate
    '''

    def new_data(self, theta, u):
        self.theta = theta
        self.x1 = np.sin(theta)
        self.y1 = np.cos(theta)
        self.u = u

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2))
        self.ax.grid()
        self.ax.axis('equal')
        plt.axis([-2, 2, -2, 2])
        self.line, = self.ax.plot([], [], 'o-', lw=2)
        self.time_template = 'time = %.1fs \nangle = %.2frad\ncontrol = %.2f'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)

    def init(self):
        self.line.set_data([], [])
        self.time_text.set_text('')
        return self.line, self.time_text

    def _update(self, i):
        thisx = [0, self.x1[i]]
        thisy = [0, self.y1[i]]
        self.line.set_data(thisx, thisy)
        self.time_text.set_text(self.time_template % (self.t[i], self.theta[i], self.u[i]))
        return self.line, self.time_text

    def start(self, path, init_theta, init_v, sigma=None):
        print('Starting Animation')
        print()
        # Set up plot to call animate() function
        print(self.t.shape[0])
        self.ani = FuncAnimation(self.fig, self._update, frames=range(self.t.shape[0]), interval=25, blit=True,
                                 init_func=self.init, repeat=False)
        self.ani.save(path + '/animation' + str((init_theta, init_v, sigma)) + '.gif', writer='imagemagick',
                      fps=10)


class Planner:
    '''
    Planner class for RL- value iteration and policy iteration
    '''

    def __init__(self, env, a=9.8, b=1, sigma=0.1 * np.eye(2), k=1, r=1, gamma=0.9, algorithm=None):
        self.env = env
        self.value = np.random.randn(self.env.n1, self.env.n2)
        self.policy = np.zeros((self.env.n1, self.env.n2))
        x1 = np.linspace(-np.pi, np.pi, self.env.n1)
        x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
        self.inte_policy = interpolate.interp2d(x1, x2, self.policy.T, kind='cubic')
        self.a = a
        self.b = b
        self.sigma = sigma
        self.k = k
        self.r = r
        self.gamma = gamma

    def reInterpolate(self):
        x1 = np.linspace(-np.pi, np.pi, self.env.n1)
        x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
        self.inte_policy = interpolate.interp2d(x1, x2, self.policy.T, kind='cubic')

    def motion(self, x, u):
        '''
        motion model: dx = f(x,u)dt + sigma*dw
        '''
        mu = x + self.f(x, u) * self.env.dt
        mu = np.array([wrap(mu[0]), min(max(mu[1], -self.env.vmax), self.env.vmax)])
        cov = self.sigma * self.env.dt
        x1 = np.linspace(-np.pi, np.pi, self.env.n1)
        x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
        py, px = np.meshgrid(x2, x1)
        x_mu0 = np.stack([px, py - mu[1]], axis=2)
        tmp = x_mu0 @ np.linalg.inv(cov)
        pf = np.exp(-1 / 2 * np.sum(np.multiply(tmp, x_mu0), axis=2))
        pf = np.roll(pf, int(mu[0] / self.env.thetamesh), axis=0)

        return pf / (np.sum(pf)), 1 - np.exp(self.k * np.cos(x[0]) - self.k) + self.r / 2 * u ** 2

    def f(self, x, u):
        '''
        function f, motion without guassian
        '''
        return np.array([x[1], self.a * np.sin(x[0]) - self.b * x[1] + u])

    def vec_motion(self, x, u):
        '''
        @x : shape (n1,n2)
        vectorized motion model: dx = f(x,u)dt + sigma*dw
        '''
        mu = x + self.vec_f(x, u) * self.env.dt  # (n1,n2,2)
        cov = self.sigma * self.env.dt
        x1 = np.linspace(-np.pi, np.pi, self.env.n1)
        x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
        py, px = np.meshgrid(x2, x1)  # (n1,n2)
        mu = mu[None, None, :, :, :]  # (1,1,n1,n2,2)
        x_mu = np.stack([px[:, :, None, None] - mu[:, :, :, :, 0], px[:, :, None, None] - mu[:, :, :, :, 1]], axis=4)

        tmp = x_mu @ np.linalg.inv(cov)
        pf = np.exp(-1 / 2 * np.sum(np.multiply(tmp, x_mu), axis=-1))  # (n1,n2,n1,n2)
        pf = np.divide(pf, np.sum(pf, axis=(0, 1), keepdims=True))

        cost = 1 - np.exp(self.k * np.cos(x[:, :, 0]) - self.k) + self.r / 2 * u ** 2
        return pf, cost

    def vec_f(self, x, u):
        '''
        vectorized function f, motion without guassian
        '''
        return np.stack([x[:, :, 1], self.a * np.sin(x[:, :, 0]) - self.b * x[:, :, 1] + u], axis=-1)

    def vec_iteration(self, algorithm='VI', thres=1e-5):
        while True:
            value_old = np.copy(self.value)
            v_list = []
            for u in np.linspace(-self.env.umax, self.env.umax, self.env.nu):
                x1 = np.linspace(-np.pi, np.pi, self.env.n1)
                x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
                x2m, x1m = np.meshgrid(x2, x1)
                x = np.stack([x1m, x2m], axis=-1)
                pf, cost = self.vec_motion(x, u)
                vu = cost + self.gamma * np.sum(np.multiply(pf, self.value[:, :, None, None]), axis=(0, 1))
                v_list.append(vu)
            v_list = np.stack(v_list, axis=2)
            self.value = np.min(v_list, axis=2)
            print(np.max(np.abs(self.value - value_old)))
            if np.max(np.abs(self.value - value_old)) < thres:
                break

    def policy_improvement(self):
        '''
        policy pi improvement
        '''
        v_list = []
        for u in np.linspace(-self.env.umax, self.env.umax, self.env.nu):
            vu = np.zeros(self.value.shape)
            for i, x1 in enumerate(np.linspace(-np.pi, np.pi, self.env.n1)):
                for j, x2 in enumerate(np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)):
                    x = np.array([x1, x2])
                    pf, cost = self.motion(x, u)
                    vu[i, j] = cost + self.gamma * np.sum(np.multiply(pf, self.value))
            v_list.append(vu)
        v_list = np.stack(v_list, axis=2)
        self.policy = np.argmin(v_list, axis=2) * self.env.umesh - self.env.umax
        x1 = np.linspace(-np.pi, np.pi, self.env.n1)
        x2 = np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)
        self.inte_policy = interpolate.interp2d(x1, x2, self.policy.T, kind='cubic')

    def iteration(self, algorithm='VI', thres=0.5):
        kk = 0
        if algorithm == 'VI':
            while True and kk < 100:
                value_old = np.copy(self.value)
                v_list = []
                for u in np.linspace(-self.env.umax, self.env.umax, self.env.nu):
                    vu = np.zeros(self.value.shape)
                    for i, x1 in enumerate(np.linspace(-np.pi, np.pi, self.env.n1)):
                        for j, x2 in enumerate(np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)):
                            x = np.array([x1, x2])
                            pf, cost = self.motion(x, u)
                            vu[i, j] = cost + self.gamma * np.sum(np.multiply(pf, self.value))
                    v_list.append(vu)
                v_list = np.stack(v_list, axis=2)
                self.value = np.min(v_list, axis=2)
                print(kk, np.max(np.abs(self.value - value_old)))
                kk += 1
                np.save(path + 'value' + str(kk).zfill(3) + '.npy', self.value)
                if np.max(np.abs(self.value - value_old)) < thres:
                    break
        if algorithm == 'PI':
            while True and kk < 100:
                value_old = np.copy(self.value)
                # policy evaluation
                while True:
                    vu = np.copy(self.value)
                    for i, x1 in enumerate(np.linspace(-np.pi, np.pi, self.env.n1)):
                        for j, x2 in enumerate(np.linspace(-self.env.vmax, self.env.vmax, self.env.n2)):
                            x = np.array([x1, x2])
                            pf, cost = self.motion(x, self.get_policy(x1, x2))
                            vu[i, j] = cost + self.gamma * np.sum(np.multiply(pf, self.value))
                    if np.max(np.abs(self.value - vu)) < thres:
                        break
                    self.value = vu
                self.policy_improvement()
                print(kk, np.max(np.abs(self.value - value_old)))
                kk += 1
                np.save(path + 'value' + str(kk).zfill(3) + '.npy', self.value)
                if np.max(np.abs(self.value - value_old)) < thres:
                    break
        if kk == 100:
            print('converge failure!')
            np.save(path + 'failure.npy', np.zeros((1, 1)))

    def state_to_ind(self, state, u):
        i1 = int((state[0] + np.pi) / self.env.thetamesh) % self.env.n1
        i2 = int((state[1] + self.env.vmax) / self.env.vmesh)
        i3 = int((u + self.env.umax) / self.env.umesh)
        return (self.env.n1 + i1) % self.env.n1, i2 % self.env.n2, i3

    def ind_to_state(self, i1=0, i2=0, i3=0):
        return i1 * self.env.thetamesh - np.pi, i2 * self.env.vmesh - self.env.vmax, i3 * self.env.umesh - self.env.umax

    def get_policy(self, x1, x2):
        return self.inte_policy(x1, x2)

    def sample(self, ti=0, vi=0, ui=0):
        theta = [ti]
        v = [vi]
        u = [ui]
        sum_cost = 0
        for i in range(self.env.t.shape[0]):
            x1, x2 = theta[-1], v[-1]
            control = self.get_policy(x1, x2)
            x = np.array([x1, x2])
            _, cost = self.motion(x, control)
            sum_cost += cost
            x = x + self.f(x, control) * self.env.dt + (self.sigma @ np.random.randn(2, 1)).T * self.env.dt
            x1, x2 = wrap(x[0, 0]), max(min(x[0, 1], self.env.vmax), -self.env.vmax)
            theta.append(x1)
            v.append(x2)
            u.append(control)
        print('totol cost:', sum_cost)
        return np.array(theta), v, u, sum_cost


def choice(pf):
    n1, n2 = pf.shape
    indexes = np.arange(n1 * n2)
    index = np.random.choice(indexes, p=pf.reshape(-1) / np.sum(pf))
    return index // n2, index % n2


def wrap(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle


def generate_hyperlist():
    hyperlist = []
    for sigma in [2,3,4,5,6,7,8,10]:
        for b in [0.1]:
            for r in [0.1]:
                for gamma in [0.95]:
                    for alg in ['VI']:
                        env_hyperparameters = {
                            'dt': 0.1,
                            'n1': 180,
                            'n2': 40,
                            'nu': 40
                        }
                        planner_hyperparameters = {
                            'a': 1,  # mass * gravity
                            'b': b,  # damping coefficient
                            'sigma': sigma,  # noise level
                            'k': 3,  # cost coefficient for height
                            'r': r,  # cost coefficient for fuel(energy)
                            'gamma': gamma,  # forgetting rate
                            'algorithm': alg
                        }
                        hyperlist.append([env_hyperparameters, planner_hyperparameters])
    return hyperlist


def main(env_hyperparameters=None, planner_hyperparameters=None):
    planner_hyperparameters['sigma'] = planner_hyperparameters['sigma'] * np.eye(2)

    animation = EnvAnimate(**env_hyperparameters)
    planner = Planner(animation, **planner_hyperparameters)

    start = time.time()
    planner.iteration(algorithm=planner_hyperparameters['algorithm'], thres=1e-3)
    planner.policy_improvement()
    tol_time = time.time() - start
    # planner.value = np.load('value.npy')
    # planner.policy = np.load('policy.npy')
    # planner.reInterpolate()

    x = np.linspace(-np.pi, np.pi, animation.n1)
    y = np.linspace(-animation.vmax, animation.vmax, animation.n2)

    X, Y = np.meshgrid(x, y)
    plt.figure()
    plt.pcolor(X, Y, planner.value.T)
    plt.colorbar()
    plt.savefig(path + './value.png')
    plt.xlabel('angle')
    plt.ylabel('velocity')
    plt.close()

    plt.figure()
    plt.pcolor(X, Y, planner.policy.T)
    plt.colorbar()
    plt.xlabel('angle')
    plt.ylabel('velocity')
    plt.savefig(path + '/policy.png')
    plt.close()

    np.save('policy.npy', planner.policy)
    np.save('value.npy', planner.value)
    tol_cost = []
    for init_theta, init_v in [(-np.pi, 0), (-np.pi / 2, 0), (-np.pi / 10, 0), (0, 0), (0, np.pi / 10)]:
        theta, v, u, cost = planner.sample(init_theta, init_v, 0)
        tol_cost.append(cost)
        animation.new_data(theta, u)
        animation.start(path, init_theta, init_v)
    for init_theta, init_v in [(0, 0)]:
        for sigma in [0.01, 0.1, 1, 2, 4, 8, 10]:
            planner.sigma = sigma * np.eye(2)
            theta, v, u, cost = planner.sample(init_theta, init_v, 0)
            tol_cost.append(cost)
            animation.new_data(theta, u)
            animation.start(path, init_theta, init_v, sigma)
    print('time elapse:', tol_time, '   cost:', tol_cost)
    pickle.dump([env_hyperparameters, planner_hyperparameters, planner.policy, planner.value, tol_cost, tol_time],
                open(path + 'data.pickle', 'wb'))


if __name__ == '__main__':
    hyperlist = generate_hyperlist()
    for env_hyperparameters, planner_hyperparameters in hyperlist:
        print(planner_hyperparameters)
        print(env_hyperparameters)
        path = './new_result/' + str(env_hyperparameters.values()) + str(planner_hyperparameters.values()) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            continue
        main(env_hyperparameters, planner_hyperparameters)
