import numpy as np
import time
import matplotlib.pyplot as plt;

import pickle
import os

plt.ion()
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import RobotPlanner


def tic():
    return time.time()


def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))


def load_map(fname):
    mapdata = np.loadtxt(fname, dtype={'names': ('type', 'xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b'), \
                                       'formats': ('S8', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f')})
    blockIdx = mapdata['type'] == b'block'
    boundary = mapdata[~blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']].view(('<f4', 9))
    blocks = mapdata[blockIdx][['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax', 'r', 'g', 'b']].view(('<f4', 9))
    return boundary, blocks


def draw_map(boundary, blocks, start, goal):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hb = draw_block_list(ax, blocks)
    hs = ax.plot(start[0:1], start[1:2], start[2:], 'ro', markersize=7, markeredgecolor='k')
    hg = ax.plot(goal[0:1], goal[1:2], goal[2:], 'go', markersize=7, markeredgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(boundary[0, 0], boundary[0, 3])
    ax.set_ylim(boundary[0, 1], boundary[0, 4])
    ax.set_zlim(boundary[0, 2], boundary[0, 5])
    return fig, ax, hb, hs, hg


def draw_block_list(ax, blocks):
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float')
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    clr = blocks[:, 6:] / 255
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    fcl = np.zeros((6 * n, 3))
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
        fcl[k * 6:(k + 1) * 6, :] = clr[k, :]

    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=0.25, linewidths=1, edgecolors='k')
        pc.set_facecolor(fcl)
        h = ax.add_collection3d(pc)
        return h


def runtest(mapfile, start, goal):
    global algorithm, N, eps, num_of_tree, reconnect, max_num_of_move_perexpand, inline, RRT_connect, savefig, fig_path, verbose
    print(algorithm)
    st = time.time()
    # Instantiate a robot planner
    boundary, blocks = load_map(mapfile)
    RP = RobotPlanner.RobotPlanner(boundary, blocks, savefig, fig_path)

    # Display the environment
    if verbose:
        fig, ax, hb, hs, hg = draw_map(boundary, blocks, start, goal)

    # args = tuple([fig, ax, hb, hs, hg])
    # Main loop
    robotpos = np.copy(start)
    numofmoves = 0
    move_list = []
    success = True
    # plot for collision testing
    # fig2, ax2, hb2, hs2, hg2 = draw_map(boundary, blocks, start, goal)
    # args = tuple([fig2, ax2, hb2, hs2, hg2])
    args = None
    distance = 0
    route = []
    while True:
        if time.time() - st > 1800:
            return False, numofmoves, distance
        # Call the robot planner
        t0 = tic()
        if algorithm == 'greedy':
            newrobotpos = RP.plan(robotpos, goal)
        if algorithm == 'RTAA':
            newrobotpos = RP.plan_RTAA(robotpos, goal, N, eps, max_num_of_move_perexpand)
        if algorithm == 'Vis-graph':
            newrobotpos = RP.plan_vis(robotpos, goal, eps, args)
        if algorithm == 'RRT':
            newrobotpos = RP.plan_RRT(robotpos, goal, eps, pg)
        if algorithm == 'BiRRT':
            newrobotpos = RP.plan_BiRRT(robotpos, goal, eps, RRT_connect, args)
        if algorithm == 'nRRT':
            newrobotpos = RP.plan_nRRT(robotpos, goal, eps, args, num_of_tree, reconnect, inline, RRT_connect)

        route.append(newrobotpos)
        movetime = max(1, np.ceil((tic() - t0) / 2.0))
        # print('move time: %d' % movetime)

        # See if the planner was done on time
        if movetime > 1:
            print('time exceeded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            newrobotpos = robotpos - 0.5 + np.random.rand(3)

        # Check if the commanded position is valid
        # print(np.linalg.norm(newrobotpos - robotpos))
        distance += np.linalg.norm(newrobotpos - robotpos)
        if sum((newrobotpos - robotpos) ** 2) > 1:
            print('ERROR: the robot cannot move so fast!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            success = False
        if (newrobotpos[0] < boundary[0, 0] or newrobotpos[0] > boundary[0, 3] or \
                newrobotpos[1] < boundary[0, 1] or newrobotpos[1] > boundary[0, 4] or \
                newrobotpos[2] < boundary[0, 2] or newrobotpos[2] > boundary[0, 5]):
            print('ERROR: out-of-map robot position commanded!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            success = False
        for k in range(blocks.shape[0]):
            if (newrobotpos[0] > blocks[k, 0] and newrobotpos[0] < blocks[k, 3] and \
                    newrobotpos[1] > blocks[k, 1] and newrobotpos[1] < blocks[k, 4] and \
                    newrobotpos[2] > blocks[k, 2] and newrobotpos[2] < blocks[k, 5]):
                print('ERROR: collision... BOOM, BAAM, BLAAM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
                success = False
                break
        if success is False:
            break

        # Make the move
        robotpos = newrobotpos
        numofmoves += 1
        move_list.append(newrobotpos)
        # Update plot
        if verbose:
            hs[0].set_xdata(robotpos[0])
            hs[0].set_ydata(robotpos[1])
            hs[0].set_3d_properties(robotpos[2])
            fig.canvas.flush_events()
            plt.show()

        # Check if the goal is reached
        if sum((robotpos - goal) ** 2) <= 0.1:
            break
    print('distance:', distance)
    total_time = time.time() - st
    print('total time usage:', total_time)

    data = [[algorithm, max_num_of_move_perexpand, N, eps, pg, num_of_tree, RRT_connect, reconnect, inline],
            [success, numofmoves, distance, total_time]]
    fig1, ax1, hb1, hs1, hg1 = draw_map(boundary, blocks, start, goal)
    for i, pos in enumerate(move_list):
        ax1.plot(pos[0:1], pos[1:2], pos[2:], 'ro', markersize=3, markeredgecolor='k')
    fig1.canvas.flush_events()
    plt.show()

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    pickle.dump(fig1, open(fig_path + 'path_fig.pickle', 'wb'))
    pickle.dump(data, open(fig_path + 'data.pickle', 'wb'))
    pickle.dump(route, open(fig_path + 'route.pickle', 'wb'))
    plt.close()

    return success, numofmoves, distance


def test_single_cube():
    start = np.array([2.3, 2.3, 1.3])
    goal = np.array([7.0, 7.0, 6.0])
    success, numofmoves, distance = runtest('./maps/single_cube.txt', start, goal)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_maze():
    start = np.array([0.0, 0.0, 1.0])
    goal = np.array([12.0, 12.0, 5.0])
    success, numofmoves, distance = runtest('./maps/maze.txt', start, goal)
    print('Success: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_window():
    start = np.array([0.2, -4.9, 0.2])
    goal = np.array([6.0, 18.0, 3.0])
    success, numofmoves, distance = runtest('./maps/window.txt', start, goal)
    print('Success: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_tower():
    start = np.array([2.5, 4.0, 0.5])
    goal = np.array([4.0, 2.5, 19.5])
    success, numofmoves, distance = runtest('./maps/tower.txt', start, goal)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_flappy_bird():
    start = np.array([0.5, 2.5, 5.5])
    goal = np.array([19.0, 2.5, 5.5])
    success, numofmoves, distance = runtest('./maps/flappy_bird.txt', start, goal)
    print('Success: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_room():
    start = np.array([1.0, 5.0, 1.5])
    goal = np.array([9.0, 7.0, 1.5])
    success, numofmoves, distance = runtest('./maps/room.txt', start, goal)
    print('Success: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance: %f' % distance)


def test_monza():
    start = np.array([0.5, 1.0, 4.9])
    goal = np.array([3.8, 1.0, 0.1])
    success, numofmoves, distance = runtest('./maps/monza.txt', start, goal)
    print('Success: %r' % success)
    print('Number of Moves: %i' % numofmoves)
    print('Total Distance:', distance)


if __name__ == "__main__":
    algorithm = 'RTAA'
    max_num_of_move_perexpand = 10000
    N = 300
    eps = 10

    # algorithm = 'Vis-graph'
    # algorithm = 'RRT'
    # algorithm = 'BiRRT'
    # algorithm = 'nRRT'

    pg = 0.05  # prob goal for RRT
    num_of_tree = 10  # number of trees for nRRT
    RRT_connect = True
    reconnect = True  # reconnect the RRT graph after building
    inline = True
    savefig = True
    verbose = False
    heuristic = 'abs'
    for N, eps, max_num_of_move_perexpand in [(10000, 1, 10000),(10000, 3, 10000), (3000, 1, 10000), (3000, 3, 10000),
                                              (3000, 5, 10000), (3000, 10, 10000),
                                              (1000, 1, 10000), (1000, 3, 10000),
                                              (1000, 5, 10000), (1000, 10, 10000),
                                              (300, 1, 10000), (300, 2, 10000),
                                              (300, 3, 10000), (300, 5, 10000),
                                              (300, 10, 10000), (30, 1, 10000), (30, 3, 10000),
                                              (30, 5, 10000), (30, 10, 10000)]:
        fig_path = './result/' + algorithm + '/' + str(
            [N, eps, max_num_of_move_perexpand, heuristic]) + '/test_single_cube/'
        print(fig_path)
        test_single_cube()

        fig_path = './result/' + algorithm + '/' + str(
            [N, eps, max_num_of_move_perexpand, heuristic]) + '/test_flappy_bird/'
        print(fig_path)
        test_flappy_bird()

        fig_path = './result/' + algorithm + '/' + str([N, eps, max_num_of_move_perexpand, heuristic]) + '/test_monza/'
        print(fig_path)
        test_monza()

        fig_path = './result/' + algorithm + '/' + str([N, eps, max_num_of_move_perexpand, heuristic]) + '/test_window/'
        print(fig_path)
        test_window()

        fig_path = './result/' + algorithm + '/' + str([N, eps, max_num_of_move_perexpand, heuristic]) + '/test_tower/'
        print(fig_path)
        test_tower()

        fig_path = './result/' + algorithm + '/' + str([N, eps, max_num_of_move_perexpand, heuristic]) + '/test_room/'
        print(fig_path)
        test_room()

        fig_path = './result/' + algorithm + '/' + str([N, eps, max_num_of_move_perexpand, heuristic]) + '/test_maze/'
        print(fig_path)
        test_maze()

    algorithm = 'Vis-graph'
    for eps in [1, 2, 3, 5, 10]:
        fig_path = './result/' + algorithm + '/' + str(
            [eps, heuristic]) + '/test_single_cube/'
        print(fig_path)
        test_single_cube()

        fig_path = './result/' + algorithm + '/' + str(
            [eps, heuristic]) + '/test_flappy_bird/'
        print(fig_path)
        test_flappy_bird()

        fig_path = './result/' + algorithm + '/' + str([eps, heuristic]) + '/test_monza/'
        print(fig_path)
        test_monza()

        fig_path = './result/' + algorithm + '/' + str([eps, heuristic]) + '/test_window/'
        print(fig_path)
        test_window()

        fig_path = './result/' + algorithm + '/' + str([eps, heuristic]) + '/test_tower/'
        print(fig_path)
        test_tower()

        fig_path = './result/' + algorithm + '/' + str([eps, heuristic]) + '/test_room/'
        print(fig_path)
        test_room()

        fig_path = './result/' + algorithm + '/' + str([eps, heuristic]) + '/test_maze/'
        print(fig_path)
        test_maze()
    algorithm = 'RRT'
    eps = 1
    for pg in [0, 0.01, 0.05, 0.1, 0.2]:
        fig_path = './result/' + algorithm + '/' + str(
            [pg, heuristic]) + '/test_single_cube/'
        print(fig_path)
        test_single_cube()

        fig_path = './result/' + algorithm + '/' + str(
            [pg, heuristic]) + '/test_flappy_bird/'
        print(fig_path)
        test_flappy_bird()

        fig_path = './result/' + algorithm + '/' + str([pg, heuristic]) + '/test_window/'
        print(fig_path)
        test_window()

        fig_path = './result/' + algorithm + '/' + str([pg, heuristic]) + '/test_tower/'
        print(fig_path)
        test_tower()

        fig_path = './result/' + algorithm + '/' + str([pg, heuristic]) + '/test_room/'
        print(fig_path)
        test_room()

    algorithm = 'BiRRT'
    eps = 1
    for RRT_connect in [True, False]:
        fig_path = './result/' + algorithm + '/' + str(
            [RRT_connect, heuristic]) + '/test_single_cube/'
        print(fig_path)
        test_single_cube()

        fig_path = './result/' + algorithm + '/' + str(
            [RRT_connect, heuristic]) + '/test_flappy_bird/'
        print(fig_path)
        test_flappy_bird()

        fig_path = './result/' + algorithm + '/' + str([RRT_connect, heuristic]) + '/test_monza/'
        print(fig_path)
        test_monza()

        fig_path = './result/' + algorithm + '/' + str([RRT_connect, heuristic]) + '/test_window/'
        print(fig_path)
        test_window()

        fig_path = './result/' + algorithm + '/' + str([RRT_connect, heuristic]) + '/test_tower/'
        print(fig_path)
        test_tower()

        fig_path = './result/' + algorithm + '/' + str([RRT_connect, heuristic]) + '/test_room/'
        print(fig_path)
        test_room()

        fig_path = './result/' + algorithm + '/' + str([RRT_connect, heuristic]) + '/test_maze/'
        print(fig_path)
        test_maze()

    algorithm = 'nRRT'
    for num_of_tree, RRT_connect, reconnect in  [(5, True, False), (10, True, False),
                                                (20, True, False), (50, True, False),
                                                # number of trees
                                                (8, True, True), (8, True, False),
                                                (20, True, True), (20, True, False)
                                                # path of Reconnect
                                                ]:
        fig_path = './result/' + algorithm + '/' + str(
            [num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_single_cube/'
        print(fig_path)
        test_single_cube()

        fig_path = './result/' + algorithm + '/' + str(
            [num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_flappy_bird/'
        print(fig_path)
        test_flappy_bird()

        fig_path = './result/' + algorithm + '/' + str(
            [num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_monza/'
        print(fig_path)
        test_monza()

        fig_path = './result/' + algorithm + '/' + str(
            [num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_window/'
        print(fig_path)
        test_window()

        fig_path = './result/' + algorithm + '/' + str(
            [num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_tower/'
        print(fig_path)
        test_tower()

        fig_path = './result/' + algorithm + '/' + str([num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_room/'
        print(fig_path)
        test_room()

        fig_path = './result/' + algorithm + '/' + str([num_of_tree, RRT_connect, reconnect, heuristic]) + '/test_maze/'
        print(fig_path)
        test_maze()
