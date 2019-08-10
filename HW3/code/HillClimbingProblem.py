import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle


class Planner:
    '''
    Initialization of all necessary variables to generate a policy:
        discretized state space
        control space
        discount factor
        learning rate
        greedy probability (if applicable)
    '''

    def __init__(self, env, hx=0.1, hy=0.01, gamma=0.9, lr=0.5, T=400):
        self.box_high = env.observation_space.high
        self.box_low = env.observation_space.low
        self.size_x = int(abs(self.box_high[0] - self.box_low[0]) / hx)
        self.size_y = int(abs(self.box_high[1] - self.box_low[1]) / hy)
        self.hx = hx
        self.hy = hy
        self.gamma = gamma
        self.lr = lr
        self.Q = np.random.randn(self.size_x, self.size_y, 3)
        # print(self.Q.shape)
        # self.Q = np.zeros((self.size_x, self.size_y, 3))
        self.T = T

    '''
    Learn and return a policy via model-free policy iteration.
    '''

    def __call__(self, mc=False, on=True, eps_greedy=False, epsilon=0.1):
        return self._td_policy_iter(on, eps_greedy, epsilon)

    '''
    TO BE IMPLEMENT
    TD Policy Iteration
    Flags: on : on vs. off policy learning
    Returns: policy that minimizes Q wrt to controls
    '''

    def _td_policy_iter(self, on=True, eps_greedy=False, epsilon=0.1):
        policy = np.argmin(self.Q, axis=2)
        if eps_greedy:
            greedy_idx = np.random.uniform(size=(self.size_x, self.size_y)) < epsilon
            policy = np.where(greedy_idx, np.random.randint(0, 3, size=(self.size_x, self.size_y)), policy)

        return policy

    '''
    Sample trajectory based on a policy
    '''

    def rollout(self, env, policy=None, render=False, path=None):
        traj = []
        t = 0
        done = False
        c_state = env.reset()
        if policy is None:
            while not done and t < self.T:
                action = env.action_space.sample()
                if render:
                    env.render()
                n_state, reward, done, _ = env.step(action)

                traj.append((c_state, action, reward))
                c_state = n_state
                t += 1

            env.close()
            return traj

        else:
            while not done and t < self.T:
                action = policy[self.state_to_idx(c_state)]
                if render:
                    if path == None:
                        env.render()
                    else:
                        env.render()
                n_state, reward, done, _ = env.step(action)

                if n_state[0] == 0.6:
                    # print('out')
                    break
                traj.append((c_state, action, reward))
                c_state = n_state
                t += 1

            env.close()
            return traj

    def update_Q(self, traj, on=True):
        for i in range(len(traj) - 1):
            tmp_state, action, reward = traj[i]
            c_state = self.state_to_idx(tmp_state)
            tmp_state, n_action, _ = traj[i + 1]
            n_state = self.state_to_idx(tmp_state)
            if on:
                self.Q[c_state[0], c_state[1], action] = self.Q[c_state[0], c_state[1], action] + self.lr * (
                        -reward + self.gamma * self.Q[n_state[0], n_state[1], n_action] - self.Q[
                    c_state[0], c_state[1], action])
            else:
                self.Q[c_state[0], c_state[1], action] = self.Q[c_state[0], c_state[1], action] + self.lr * (
                        -reward + self.gamma * np.min(self.Q[n_state[0], n_state[1], :]) - self.Q[
                    c_state[0], c_state[1], action])

    def state_to_idx(self, state):
        # print(state[0],self.box_low[0])
        # print()
        return int((state[0] - self.box_low[0]) / self.hx), int((state[1] - self.box_low[1]) / self.hy)

    def traj_analysis(self, traj):
        '''
        analysis the statistical of trajectory
        '''
        tol_reward = 0
        for c_state, action, reward in traj:
            tol_reward += reward
        return len(traj) == self.T, tol_reward, len(traj)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def generate_hyper_list():
    hyper = []
    for alpha in [0.01, 0.1, 0.5, 0.9]:
        for gamma in [0.8, 0.9, 0.99]:
            for eps_decay in [0, 0.8, 0.9, 0.99, 0.95]:
                for num_episode in [1000, 3000, 10000]:
                    for on in [True, False]:
                        hyper.append([0.06, 0.005, gamma, alpha, 200, eps_decay, num_episode, 0.9, on])
    return hyper


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    hyper_list = generate_hyper_list()
    # hyper_list = [[0.06, 0.005, 0.9, 0.1, 200, 0, 10000, 1, False]]

    for hx, hy, gamma, lr, T, decay_ratio, num_episode, eps_start, on in hyper_list:
        hyperparameters = {'hx': hx,
                           'hy': hy,
                           'gamma': gamma,
                           'lr': lr,
                           'T': T,
                           'decay_ratio': decay_ratio,
                           'num_episode': num_episode,
                           'eps_start': eps_start,
                           'on': on
                           }
        print(hyperparameters)
        path = './result/' + str(hyperparameters.values()) + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        planner = Planner(env, hx=hx, hy=hy, gamma=gamma, lr=lr, T=T)

        # # training
        # reward_list = []
        # step_list = []
        # eps_list = []
        # eps = hyperparameters['eps_start']
        # for i in range(hyperparameters['num_episode']):
        #     if i and i % 300 == 0:
        #         eps *= hyperparameters['decay_ratio']
        #     if hyperparameters['decay_ratio'] == 0:
        #         eps = 0
        #     eps_list.append(eps)
        #     # print(eps)
        #     Q_old = np.copy(planner.Q)
        #     policy = planner(eps_greedy=True, epsilon=eps)
        #     traj = planner.rollout(env, policy=policy, render=False)
        #     planner.update_Q(traj, on=on)
        #     success, tol_reward, step = planner.traj_analysis(traj)
        #     reward_list.append(tol_reward)
        #     step_list.append(step)
        #
        #     if i % 300 == 0:
        #         plt.figure()
        #         fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(7, 3))
        #         for t in range(3):
        #             value = planner.Q[:, :, t]
        #             im = axes[t].imshow(value)
        #             fig.colorbar(im, ax=axes[t])
        #         im = axes[3].imshow(policy)
        #         fig.colorbar(im, ax=axes[3])
        #         fig.suptitle(hyperparameters, fontsize=7)
        #         plt.savefig(path + 'Q' + str(i) + 'off.png')
        #         # np.save(path + 'Q.npy', planner.Q)
        #         plt.close()
        #
        #     if i >= 500 and i % 500 == 0:
        #         # print('plotting')
        #         plt.close('all')
        #         plt.figure()
        #         plt.plot(moving_average(reward_list, 500))
        #         plt.title(hyperparameters)
        #         plt.ylabel('reward')
        #         plt.savefig(path + 'reward.png')
        #
        #         plt.figure()
        #         plt.title(hyperparameters)
        #         plt.ylabel('steps')
        #         plt.plot(moving_average(step_list, 500))
        #         plt.savefig(path + 'step.png')
        #
        #         plt.figure()
        #         plt.title(hyperparameters)
        #         plt.ylabel('eps')
        #         plt.plot(eps_list)
        #         plt.savefig(path + 'eps.png')
        #
        # policy = planner(eps_greedy=False)
        # plt.figure()
        # plt.imshow(policy)
        # np.save(path + 'policy.npy', policy)
        # plt.title(hyperparameters, fontsize=7)
        # plt.colorbar()
        # plt.savefig(path + 'policy.png')
        #
        # plt.figure()
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(7, 2))
        # for i in range(3):
        #     value = planner.Q[:, :, i]
        #     im = axes[i].imshow(value)
        #     fig.colorbar(im, ax=axes[i])
        # fig.suptitle(hyperparameters, fontsize=7)
        # plt.savefig(path + 'Q.png')
        # np.save(path + 'Q.npy', planner.Q)
        #
        # plt.close('all')

        policy = np.load(path + 'policy.npy')
        avg_len = 0
        for i in range(100):
            traj = planner.rollout(env, policy=policy, render=False, path=path)
            avg_len += len(traj)
        avg_len /= 100

        # env = gym.wrappers.Monitor(env, path, video_callable=lambda episode_id: True, force=True)
        # traj = planner.rollout(env, policy=policy, render=True, path=path)

        # pickle.dump([traj, hyperparameters, planner.Q, policy, moving_average(step_list, 500), avg_len],
        #             open(path + 'data.pickle', 'wb'))

        pickle.dump([hyperparameters, avg_len],
                    open(path + 'avg_len.pickle', 'wb'))
        # print(traj)
        print(avg_len)
        # print(len(traj))
