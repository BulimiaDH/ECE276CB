Simply run the main.py file.

I implement RTAA*, Visibility graph, RRT, BIRRT and nRRT algorithm.
In the main file, I loop over the hyperparameters and algorithm and save it as .pickle file, then use visualize.py file to visualize.
You can tune all the hyperparameters in loop:
    
    max_num_of_move_perexpand 
    N : number of expand node in RTAA*
    eps : epsilon for heuristic
    algorithm = 'Vis-graph' or 'RRT'or 'BiRRT' or 'nRRT' or 'RRTA*'
    pg :prob goal for RRT
    num_of_tree : number of trees for nRRT
    RRT_connect : change naive BiRRT and nRRT to RRT-connect
    reconnect : reconnect(rewiring) the RRT graph after building
    inline : generate initial root inline with start and goal node
    savefig : savefig as pickle file
    verbose : plot the track simutaneously
    heuristic : heuristic function , can choose 'abs' for manhattan distance, 'l2' for l2 norm