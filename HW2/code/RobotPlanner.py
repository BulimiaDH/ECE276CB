import heapq as hq
import math
import random
import time
import pickle

import matplotlib.pyplot as plt
import numpy as np


class RobotPlanner:
    __slots__ = ['boundary', 'blocks', 'h_map', 'route', 'vis_graph', 'dec', 'collision_check_step_size', 'point_list',
                 'si', 'sj', 'start_node', 'RRT_graph', 'RRT_graph_a', 'RRT_graph_b', 'args', 'nRRT_list',
                 'reconnect_index', 'finished_build', 'r', 'move_counter', 'savefig', 'fig_path', 'OPEN', 'CLOSED',
                 'st_index', 'parent', 'gvalue']

    def __init__(self, boundary, blocks, savefig, fig_path):
        self.boundary = boundary
        self.blocks = blocks
        # for all
        self.dec = 5
        self.collision_check_step_size = 0.05
        self.route = []

        # for RRTA*
        self.OPEN = []
        self.CLOSED = set()
        self.st_index = 0
        self.parent = {}
        self.gvalue = {}
        self.h_map = {}
        self.move_counter = 0

        # for vis-graph
        self.vis_graph = {}
        self.point_list = []
        self.si = 0
        self.sj = 0
        self.start_node = -1

        # for RRT
        self.RRT_graph = {}

        # for Bi-RRT
        self.RRT_graph_a = {}
        self.RRT_graph_b = {}

        # for n-RRT
        self.nRRT_list = []
        self.reconnect_index = 0
        self.finished_build = False
        self.r = -1

        # for test plot
        self.args = None

        # code for savefig
        self.savefig = savefig
        self.fig_path = fig_path

    def plan(self, start, goal):
        # for now greedily move towards the goal
        newrobotpos = np.copy(start)

        numofdirs = 26
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / np.sqrt(np.sum(dR ** 2, axis=0)) / 2.0

        mindisttogoal = 1000000
        for k in range(numofdirs):
            newrp = start + dR[:, k]

            # Check if this direction is valid
            if (newrp[0] < self.boundary[0, 0] or newrp[0] > self.boundary[0, 3] or \
                    newrp[1] < self.boundary[0, 1] or newrp[1] > self.boundary[0, 4] or \
                    newrp[2] < self.boundary[0, 2] or newrp[2] > self.boundary[0, 5]):
                continue

            valid = True
            for k in range(self.blocks.shape[0]):
                if (newrp[0] > self.blocks[k, 0] and newrp[0] < self.blocks[k, 3] and \
                        newrp[1] > self.blocks[k, 1] and newrp[1] < self.blocks[k, 4] and \
                        newrp[2] > self.blocks[k, 2] and newrp[2] < self.blocks[k, 5]):
                    valid = False
                    break
            if not valid:
                break

            # Update newrobotpos
            disttogoal = sum((newrp - goal) ** 2)
            if (disttogoal < mindisttogoal):
                mindisttogoal = disttogoal
                newrobotpos = newrp

        return newrobotpos

    def plan_RTAA(self, start, goal, N, eps, max_move, args=None):
        '''
        RTA A* implementation of planning
        :param start: start node
        :param goal: goal node
        :param N: number of expansion at each time step
        :return: new position of robot
        '''

        if isinstance(self.start_node, int):
            self.start_node = start
        if len(self.route) == 0 or self.move_counter >= max_move:
            self.route = []
            finished = self.replan_RTAA(start, goal, N, eps)
            if not finished:
                return self.random_move(start)
        self.move_counter += 1
        return self.route.pop()

    def replan_RTAA(self, start, goal, N, eps, args=None):
        '''
        every N step, need to re-plan
         :param start: start node
        :param goal: goal node
        :param N: number of expansion at each time step
        :return: new position of robot
        '''

        st = time.time()
        # 3D direction([-1,0,1]) for x,y,z(3*3*3), but remove (0,0,0)
        newrobotpos = np.copy(start)
        dec = self.dec
        gamma = 3.0

        if self.st_index == 0:
            self.OPEN = []
            self.CLOSED = set()
            self.CLOSED.add(tuple(np.around(newrobotpos, dec)))
            self.parent = {}
            self.gvalue = {}
            self.gvalue[tuple(np.around(newrobotpos, dec))] = 0
        numofdirs = 26
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / gamma

        f_j = 0
        for n in range(self.st_index, N):
            if time.time() - st > 1.7:
                self.st_index = n
                return False
            # collision checking
            for k in range(numofdirs):
                newrp = newrobotpos + dR[:, k]
                if tuple(newrp) in self.CLOSED:
                    continue
                # Check if this direction is valid
                if self.boundary_checker(newrp, name='equal'):
                    continue
                valid = True

                for kk in range(self.blocks.shape[0]):
                    if self.collision_check(newrobotpos, newrp, kk, name='step_equal'):
                        valid = False
                        break
                if not valid:
                    continue
                if (tuple(np.around(newrp, dec)) not in self.gvalue) or (
                        self.gvalue[tuple(np.around(newrp, dec))] > self.gvalue[
                    tuple(np.around(newrobotpos, dec))] + np.linalg.norm(dR[:, k])):
                    self.gvalue[tuple(np.around(newrp, dec))] = self.gvalue[tuple(
                        np.around(newrobotpos, dec))] + np.linalg.norm(
                        dR[:, k])
                    self.parent[tuple(newrp)] = newrobotpos

                    if tuple(np.around(newrp, dec)) in self.h_map:
                        hq.heappush(self.OPEN, (eps * self.h_map[tuple(np.around(newrp, dec))] + \
                                                self.gvalue[tuple(np.around(newrp, dec))], tuple(newrp)))
                    else:
                        hq.heappush(self.OPEN,
                                    (eps * heuristic(newrp, goal) + self.gvalue[tuple(np.around(newrp, dec))],
                                     tuple(newrp)))

            # Update newrobotpos, might need round operation
            f_j, newrobotpos = hq.heappop(self.OPEN)
            while tuple(np.around(newrobotpos, dec)) in self.CLOSED:
                f_j, newrobotpos = hq.heappop(self.OPEN)
            self.CLOSED.add(tuple(np.around(newrobotpos, dec)))
            newrobotpos = np.array(newrobotpos)
            if sum((newrobotpos - goal) ** 2) <= 0.1:
                print('found!')
                break
            # print(f_j)
            # draw_on_canvas(newrobotpos, args)
        end_pos = newrobotpos
        while not np.allclose(end_pos, start):
            self.route.append(end_pos)
            end_pos = self.parent[tuple(end_pos)]
        # print(len(self.route))

        # update heuristic
        for c in self.CLOSED:
            self.h_map[tuple(np.around(c, dec))] = f_j - self.gvalue[c]
        self.st_index = 0
        return True

    def random_move(self, start):
        sz = 0
        return start
        # [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        # dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        # dR = np.delete(dR, 13, axis=1)
        # dR = dR * sz
        # numofdirs = 26
        # for k in range(numofdirs):
        #     newrp = start + dR[:, k]
        #     # Check if this direction is valid
        #     if self.boundary_checker(newrp):
        #         continue
        #     for kk in range(self.blocks.shape[0]):
        #         if not self.collision_check(start, newrp, kk, name='naive'):
        #             return newrp

    def plan_vis(self, start, goal, eps, args=None):
        '''
        implementation of visibility graph of A*
        :param robotpos: current position
        :param goal: goal node
        :param eps: heuristic eps
        :param args: plot args
        :return: next position
        '''
        if isinstance(self.start_node, int):
            self.start_node = start
        if len(self.route) == 0:
            finished, self.si, self.sj = self.build_graph(self.start_node, goal, self.si, self.sj)
            if not finished:
                return self.random_move(start)
            self.replan_vis_A_star(self.start_node, goal, eps)
            if not np.allclose(self.start_node, start):
                return self.start_node
        dir = self.route[-1] - start
        norm_dir = np.linalg.norm(dir)
        if norm_dir >= 0.99:
            return start + 0.99 * dir / norm_dir
        else:
            return self.route.pop()

    def build_graph(self, robotpos, goal, starti=0, startj=0):
        '''
        build visibility graph
        :return: None
        '''
        start_time = time.time()
        step_size = self.collision_check_step_size
        if len(self.point_list) == 0:
            self.point_list += [list(robotpos), list(goal)]
            for kk in range(self.blocks.shape[0]):
                b1, b2, b3, b4, b5, b6 = self.blocks[kk, 0], self.blocks[kk, 1], self.blocks[kk, 2], self.blocks[kk, 3], \
                                         self.blocks[kk, 4], self.blocks[kk, 5]
                self.point_list += [[b1, b2, b3], [b1, b2, b6], [b1, b5, b3], [b1, b5, b6], [b4, b2, b3], [b4, b2, b6],
                                    [b4, b5, b3], [b4, b5, b6]]
        for i in range(starti, len(self.point_list)):
            print(i)
            for j in range(0, len(self.point_list)):
                if j < self.sj:
                    continue
                else:
                    self.sj = 0
                if time.time() - start_time > 1.5:
                    return False, i, j
                pi = np.array(self.point_list[i])
                pj = np.array(self.point_list[j])
                dist = np.linalg.norm(pi - pj)
                if tuple(pj) in self.vis_graph:
                    if tuple([tuple(pi), dist]) in self.vis_graph[tuple(pj)]:
                        if not tuple(pi) in self.vis_graph:
                            self.vis_graph[tuple(pi)] = []
                        self.vis_graph[tuple(pi)].append(tuple([tuple(pj), dist]))
                    continue
                n = int(dist / step_size)
                # n = 2  # for middle point
                valid = True
                if self.entire_collision_check(pi, pj, n, name='step', volume_checker=True):
                    valid = False
                if valid:
                    if not tuple(pi) in self.vis_graph:
                        self.vis_graph[tuple(pi)] = []
                    self.vis_graph[tuple(pi)].append(tuple([tuple(pj), dist]))
        return True, -1, -1

    def replan_vis_A_star(self, robotpos, goal, eps):
        '''
        A_star algorithm on vis_graph
        :param robotpos:
        :param goal:
        :param N:
        :param eps:
        :return:
        '''
        # preparation
        dec = self.dec
        gamma = 3.0
        [dX, dY, dZ] = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])
        dR = np.vstack((dX.flatten(), dY.flatten(), dZ.flatten()))
        dR = np.delete(dR, 13, axis=1)
        dR = dR / gamma
        gvalue = {}
        gvalue[tuple(robotpos)] = 0
        f_j = 0

        parent = {}
        OPEN = [(heuristic(robotpos, goal), tuple(robotpos))]
        CLOSED = set()
        while len(OPEN) > 0:
            f_j, newrobotpos = hq.heappop(OPEN)
            while newrobotpos in CLOSED:
                f_j, newrobotpos = hq.heappop(OPEN)
            CLOSED.add(newrobotpos)
            newrobotpos = np.array(newrobotpos)
            if np.allclose(newrobotpos, goal):
                print('success!')
                break
            for neighbor, dist in self.vis_graph[tuple(newrobotpos)]:
                if neighbor in CLOSED:
                    continue
                if neighbor not in gvalue or (gvalue[neighbor] > gvalue[tuple(newrobotpos)] + dist):
                    gvalue[neighbor] = gvalue[tuple(newrobotpos)] + dist
                    parent[neighbor] = newrobotpos
                    hq.heappush(OPEN, (eps * heuristic(np.array(neighbor), goal) + gvalue[neighbor], neighbor))

        end_pos = goal
        while not np.allclose(end_pos, robotpos):
            self.route.append(end_pos)
            end_pos = parent[tuple(end_pos)]
        print('route length: ', len(self.route))

    # code for RRT*

    def plan_RRT(self, start, goal, eps, pg):
        if isinstance(self.start_node, int):
            self.start_node = start
        if len(self.route) == 0:
            finished = self.build_RRT(self.start_node, goal, pg)
            print('size of RRT:', len(self.RRT_graph))
            if not finished:
                return self.random_move(self.start_node)
            self.plan_RRT_A_star(self.start_node, goal, eps)
        return self.route.pop()

    def plan_RRT_A_star(self, start, goal, eps):
        gvalue = {}
        gvalue[tuple(start)] = 0
        f_j = 0
        parent = {}
        OPEN = [(heuristic(start, goal), tuple(start))]
        CLOSED = set()
        while len(OPEN) > 0:
            f_j, newrobotpos = hq.heappop(OPEN)
            while newrobotpos in CLOSED:
                f_j, newrobotpos = hq.heappop(OPEN)
            CLOSED.add(newrobotpos)
            newrobotpos = np.array(newrobotpos)
            if np.allclose(newrobotpos, goal):
                print('success!')
                break
            for neighbor, dist in self.RRT_graph[tuple(newrobotpos)]:
                if neighbor in CLOSED:
                    continue
                if neighbor not in gvalue or (gvalue[neighbor] > gvalue[tuple(newrobotpos)] + dist):
                    gvalue[neighbor] = gvalue[tuple(newrobotpos)] + dist
                    parent[neighbor] = newrobotpos
                    hq.heappush(OPEN, (eps * heuristic(np.array(neighbor), goal) + gvalue[neighbor], neighbor))

        end_pos = goal
        while not np.allclose(end_pos, start):
            self.route.append(end_pos)
            end_pos = parent[tuple(end_pos)]
        print('route length: ', len(self.route))

    def build_RRT(self, start, goal, pg):
        st = time.time()
        success = False
        if len(self.RRT_graph) == 0:
            self.RRT_graph[tuple(start)] = []
        while not success:
            if time.time() - st > 1.5:
                return False
            xrand = self.sample_free(goal, pg)
            xnearest = self.nearest(xrand)
            xnew = self.steer(xnearest, xrand)
            if np.allclose(xnew, xnearest):
                continue
            # print(xnew, np.linalg.norm(xnew - goal))
            dist = np.linalg.norm(xnearest - xnew)
            self.RRT_graph[tuple(xnew)] = [tuple([tuple(xnearest), dist])]
            self.RRT_graph[tuple(xnearest)].append(tuple([tuple(xnew), dist]))
            if np.linalg.norm(xnew - goal) < 1:
                dist = np.linalg.norm(goal - xnew)
                self.RRT_graph[tuple(goal)] = [tuple([tuple(xnew), dist])]
                self.RRT_graph[tuple(xnew)].append(tuple([tuple(goal), dist]))
                success = True
        return success

    # def plan_RRT_A_star(self, start, goal):

    def steer(self, xnearest, xrand, tol=0.99, list=False):
        '''
        return the steering function, need to return xrand if possible
        tol: maximum step
        '''
        dir = xrand - xnearest
        sz = self.collision_check_step_size
        n = int(tol / sz)
        if not list:
            xnew = xnearest
            for i in range(n):
                xnew_new = xnearest + sz * (i + 1) * dir / np.linalg.norm(dir)
                if self.entire_collision_check(None, xnew_new, name='naive_equal') or self.boundary_checker(xnew_new,
                                                                                                            name='equal'):
                    return xnew
                xnew = xnew_new
            if np.linalg.norm(dir) <= 1:
                return xrand
            else:
                return xnew
        else:
            ret_list = []
            xnew = xnearest
            while True:
                xnearest = xnew
                for i in range(n):
                    xnew_new = xnearest + sz * (i + 1) * dir / np.linalg.norm(dir)
                    if self.entire_collision_check(None, xnew_new) or self.boundary_checker(xnew_new):
                        return ret_list + [xnew]
                    xnew = xnew_new
                if np.linalg.norm(dir) <= 1:
                    return ret_list + [xrand]
                else:
                    ret_list.append(xnew)

    def nearest(self, xrand, graph=None):
        '''
        return the nearest node to the xrand in the tree
        '''
        if graph is None:
            graph = self.RRT_graph
        return np.array(min(list(graph.keys()), key=lambda x: np.linalg.norm(np.array(x) - xrand)))

    def sample_free(self, goal, pg=0.0, p_edge=0.0, inline=False):
        if not inline:
            p = np.random.uniform(0, 1)
            if p > pg and p > p_edge:
                xrand = np.zeros((3,))
                xrand[0] = np.random.uniform(self.boundary[0, 0], self.boundary[0, 3])
                xrand[1] = np.random.uniform(self.boundary[0, 1], self.boundary[0, 4])
                xrand[2] = np.random.uniform(self.boundary[0, 2], self.boundary[0, 5])
                while self.entire_collision_check(None, xrand, name='naive_equal'):
                    xrand[0] = np.random.uniform(self.boundary[0, 0], self.boundary[0, 3])
                    xrand[1] = np.random.uniform(self.boundary[0, 1], self.boundary[0, 4])
                    xrand[2] = np.random.uniform(self.boundary[0, 2], self.boundary[0, 5])
                return xrand
            else:
                if pg > 0:
                    return goal
                if p_edge > 0:
                    n_block = np.random.randint(0, self.blocks.shape[0])
                    dims = np.random.choice([0, 3], size=(3,))
                    return np.array(
                        [self.blocks[n_block, dims[0]], self.blocks[n_block, 1 + dims[1]],
                         self.blocks[n_block, 2 + dims[2]]])
        else:
            ratio = np.random.uniform(0, 1)
            new_node = ratio * (goal - self.start_node) + self.start_node
            while self.entire_collision_check(None, new_node, name='naive_equal'):
                ratio = np.random.uniform(0, 1)
                new_node = ratio * (goal - self.start_node) + self.start_node
            return new_node

    # code for Bi-RRT
    def plan_BiRRT(self, start, goal, eps, RRT_connect=False, args=None):
        if args is not None:
            self.args = args
        if isinstance(self.start_node, int):
            self.start_node = start
        if tuple(start) not in self.RRT_graph_a:
            self.RRT_graph_a[tuple(start)] = []
            self.RRT_graph_b[tuple(goal)] = []
        if len(self.route) == 0:
            finished, self.RRT_graph,_ = self.build_BiRRT(self.start_node, goal, self.RRT_graph_a, self.RRT_graph_b,
                                                        args=args,
                                                        lis=RRT_connect)
            print(finished, 'size of BiRRT:', len(self.RRT_graph_a), len(self.RRT_graph_b))
            # self.show_RRT(args)
            if not finished:
                return self.random_move(self.start_node)
            # print(self.RRT_graph)
            self.plan_RRT_A_star(self.start_node, goal, eps)
        return self.route.pop()

    def build_BiRRT(self, start, goal, graph_pointer, graph_pointer_2, args=None, lis=False):
        '''
        :param start: start node
        :param goal: goal node
        :param args: plot args for testing
        :param list: boolean, true to connect all the way to xrand
        :return:
        '''
        st = time.time()
        success = False
        while not success:
            if time.time() - st > 1.5:
                return False, None, None
            xrand = self.sample_free(goal, 0, p_edge=0.00)
            xnearest = self.nearest(xrand, graph_pointer)
            xc_list = self.steer(xnearest, xrand, list=lis)
            # print(len(xc_list))
            if not isinstance(xc_list, list):
                xc_list = [xc_list]
            for k, xc in enumerate(xc_list):
                if not np.allclose(xc, xnearest):
                    dist = np.linalg.norm(xnearest - xc)
                    graph_pointer[tuple(xc)] = [(tuple([tuple(xnearest), dist]))]
                    graph_pointer[tuple(xnearest)].append(tuple([tuple(xc), dist]))
                    if k < len(xc_list) - 1:
                        xnearest = xc
            # draw_on_ax(xc, args)
            if not np.allclose(xc, xnearest):
                xnearest_prime = self.nearest(xc, graph_pointer_2)
                xc_prime = self.steer(xnearest_prime, xc)
                if not np.allclose(xnearest_prime, xc_prime):
                    dist = np.linalg.norm(xnearest_prime - xc_prime)
                    graph_pointer_2[tuple(xc_prime)] = [tuple([tuple(xnearest_prime), dist])]
                    graph_pointer_2[tuple(xnearest_prime)].append(tuple([tuple(xc_prime), dist]))
                if np.allclose(xc_prime, xc):
                    # print(xc_prime - xc)
                    graph_pointer_2.update(graph_pointer)
                    graph_pointer_2[tuple(xc_prime)].append(tuple([tuple(xnearest_prime), dist]))
                    print('connect!')
                    return True, graph_pointer_2, graph_pointer
            if len(graph_pointer) > len(graph_pointer_2):
                tmp = graph_pointer
                graph_pointer = graph_pointer_2
                graph_pointer_2 = tmp

    def plan_nRRT(self, start, goal, eps, args, num_of_tree, reconnect=False, inline=False, RRT_connect=False):
        if args is not None:
            self.args = args
        if isinstance(self.start_node, int):
            self.start_node = start
        if len(self.route) == 0:
            if not self.finished_build:
                self.finished_build = self.build_nRRT(self.start_node, goal, args, num_of_tree, inline,
                                                      RRT_connect=RRT_connect)
                print(self.finished_build, 'size of nRRT:', len(self.nRRT_list),
                      [len(self.nRRT_list[i]) for i in range(len(self.nRRT_list))])

                if not self.finished_build:
                    return self.random_move(self.start_node)
            # print(self.RRT_graph)
            finished = True
            if reconnect:
                finished = self.reconnect('naive')
            if not finished:
                return self.random_move(self.start_node)
            st = time.time()
            self.plan_RRT_A_star(self.start_node, goal, eps)
            print('planning time use:', time.time() - st)
        pop_value = self.route.pop()
        dir = pop_value - start
        if np.linalg.norm(dir) > 0.99:
            self.route.append(pop_value)
            return start + 0.99 * dir / np.linalg.norm(dir)
        else:
            return pop_value

    def build_nRRT(self, start, goal, args, num_of_tree, inline, RRT_connect=False):
        if len(self.nRRT_list) == 0:
            self.nRRT_list = [{} for i in range(num_of_tree)]
            self.nRRT_list[0][tuple(start)] = []
            self.nRRT_list[1][tuple(goal)] = []
            for i in range(2, num_of_tree):
                tmp = self.sample_free(goal, inline=inline)
                self.nRRT_list[i][tuple(tmp)] = []
                # draw_on_ax(tmp, args)
        if len(self.nRRT_list) >= 2:
            graph_1, graph_2 = random.sample(self.nRRT_list, 2)
            # graph_1, graph_2 = sorted(self.nRRT_list, key=lambda x: len(x))[:2]
            connected, merged_graph, waste = self.build_BiRRT(start, goal, graph_1, graph_2, args, lis=RRT_connect)
            if connected:
                del self.nRRT_list[self.nRRT_list.index(waste)]
                if len(self.nRRT_list) == 1:
                    self.RRT_graph = self.nRRT_list[0]
                    return True
        return False

    def reconnect(self, name='naive'):
        '''
        reconnect node of nRRT to reduce suboptimality
        :param name: 'naive': simply connect all the node with in r*, and A* would find optimal path
        :return:
        '''
        st = time.time()
        index = self.reconnect_index
        if name == 'naive':
            if self.r == -1:
                print('starting reconnect!')
                self.r = self.compute_reconnect_r()
                print('reconnect radius: ', self.r)
            r = self.r

            for k in range(index, len(self.RRT_graph)):
                node = list(self.RRT_graph.keys())[k]
                if k % 100 == 0:
                    print(k, '/', len(self.RRT_graph))
                if time.time() - st > 1.5:
                    self.reconnect_index = k
                    return False
                reconnect_nodes = self.find_reconnect_nodes(node, r)
                for reconnect_node in reconnect_nodes:
                    dist = np.linalg.norm(np.array(node) - np.array(reconnect_node))
                    if dist == 0:
                        continue
                    if tuple([reconnect_node, dist]) not in self.RRT_graph[node]:
                        n = int(dist / self.collision_check_step_size)
                        if not self.entire_collision_check(np.array(node), np.array(reconnect_node), name='step_equal',
                                                           n=n):
                            self.RRT_graph[node].append(tuple([reconnect_node, dist]))
            return True

    def find_reconnect_nodes(self, node, r):

        return [i for i in list(self.RRT_graph.keys()) if np.linalg.norm(np.array(node) - np.array(i)) < r]

    def compute_reconnect_r(self):
        volume_obs = 0
        for i in range(self.blocks.shape[0]):
            volume_obs += volume(self.blocks[i][0:6])
        return 2 * (4 / 3) ** (1 / 3) * (volume(self.boundary[0, :6]) / volume_obs) ** (1 / 3) * (
                np.log(len(self.RRT_graph)) / len(self.RRT_graph)) ** (1 / 3)

    # code for collision checking
    def entire_collision_check(self, p1, p2, n=10, name='naive', delta=0.001, volume_checker=False):
        for kk in range(self.blocks.shape[0]):
            if self.collision_check(p1, p2, kk, n=n, name=name):
                return True
        if volume_checker:
            n_volume = n
            return self.volume_checker(p1, p2, n_volume, delta=delta)
        return False

    def volume_checker(self, p1, p2, n_volume=2, delta=0.001):
        step = (p1 - p2) / n_volume
        delta_list = delta_direction(delta, step, 8)
        for i in range(n_volume):
            count = 0
            for k, d in enumerate(delta_list):
                if self.entire_collision_check(None, p2 + i * step + d, n=1) or self.boundary_checker(
                        p2 + i * step + d, name='equal'):
                    count += 1
            if len(delta_list) - count <= 1:
                return True
        return False

    def collision_check(self, p1, p2, kk, n=10, name='naive'):
        '''

        :param newrp: newrobot position
        :param name: naive for checking if x_t+1 inside, 'step' using step for advanced checking
        :return: bool True for collision
        '''

        if name == 'naive':
            return (p2[0] > self.blocks[kk, 0]) and (p2[0] < self.blocks[kk, 3]) and (
                    p2[1] > self.blocks[kk, 1]) and (p2[1] < self.blocks[kk, 4]) and (
                           p2[2] > self.blocks[kk, 2]) and (p2[2] < self.blocks[kk, 5])

        if name == 'naive_equal':
            return (p2[0] >= self.blocks[kk, 0]) and (p2[0] <= self.blocks[kk, 3]) and (
                    p2[1] >= self.blocks[kk, 1]) and (p2[1] <= self.blocks[kk, 4]) and (
                           p2[2] >= self.blocks[kk, 2]) and (p2[2] <= self.blocks[kk, 5])

        if name == 'step':
            if n > 0:
                step = (p1 - p2) / n
                for i in range(n + 1):
                    if self.collision_check(p1, p2 + i * step, kk, name='naive'):
                        return True
            return False

        if name == 'step_equal':
            if n > 0:
                step = (p1 - p2) / n
                for i in range(n + 1):
                    if self.collision_check(p1, p2 + i * step, kk, name='naive_equal'):
                        return True
            return False

    def boundary_checker(self, newrp, name='naive'):
        if name == 'naive':
            return (newrp[0] < self.boundary[0, 0] or newrp[0] > self.boundary[0, 3] or \
                    newrp[1] < self.boundary[0, 1] or newrp[1] > self.boundary[0, 4] or \
                    newrp[2] < self.boundary[0, 2] or newrp[2] > self.boundary[0, 5])
        if name == 'equal':
            return (newrp[0] <= self.boundary[0, 0] or newrp[0] >= self.boundary[0, 3] or \
                    newrp[1] <= self.boundary[0, 1] or newrp[1] >= self.boundary[0, 4] or \
                    newrp[2] <= self.boundary[0, 2] or newrp[2] >= self.boundary[0, 5])

    def show_RRT(self, args):
        fig1, ax1, hb1, hs1, hg1 = args
        re


def delta_direction(delta, vec, numdir=6):
    theta = 2 * math.pi / numdir
    if vec[2] != 0:
        c = -(vec[0] + vec[1]) / vec[2]
        vectical_vec = np.array([1, 1, c])
    elif vec[1] != 0:
        b = -(vec[0] + vec[2]) / vec[1]
        vectical_vec = np.array([1, b, 1])
    else:
        a = -(vec[1] + vec[2]) / vec[0]
        vectical_vec = np.array([a, 1, 1])
    vectical_vec = delta * vectical_vec / np.linalg.norm(vectical_vec)
    delta_list = []
    for i in range(numdir):
        delta_list.append(np.dot(rotation_matrix(vec, i * theta), vectical_vec))
    return delta_list


def draw_on_canvas(robotpos, args):
    fig, ax, hb, hs, hg = args
    hs[0].set_xdata(robotpos[0])
    hs[0].set_ydata(robotpos[1])
    hs[0].set_3d_properties(robotpos[2])
    fig.canvas.flush_events()
    plt.show()


def draw_on_ax(start, args, color='k', size=3):
    fig, ax, hb, hs, hg = args
    hs = ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=size, markeredgecolor=color)


def heuristic(pos, goal, name='abs'):
    '''
    implementation of heuristic function
    :param pos: current position of my robot
    :param goal: goal node
    :return: heuristic value
    '''

    if name == 'l2':
        return np.linalg.norm(pos - goal)
    if name == 'abs':
        return np.sum(np.abs(pos - goal))


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def volume(obj):
    '''
    return volume of 3D blocks
    :param obj: numpy array['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax','r','g','b']
    :return: volume
    '''

    return (obj[3] - obj[0]) * (obj[4] - obj[1]) * (obj[5] - obj[2])
