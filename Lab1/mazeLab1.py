import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];
P_poison = 1/30
# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -10000
    EATEN_REWARD = -1000


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        #i,j player position. m,n minotaur position.
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for m in range(self.maze.shape[0]):
                    for n in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = (i,j,m,n);
                            map[(i,j,m,n)] = s;
                            s += 1;
        return states, map
    def getMinotaur_actions(self, state):
        # Get possible minotaur actions.
        currentM_row = self.states[state][2];
        currentM_col = self.states[state][3];
        #check if minotaur is on the border of the maze
        border_of_maze =  (currentM_row == 0) or (currentM_row== self.maze.shape[0]-1) or \
                                (currentM_col == 0) or (currentM_col== self.maze.shape[1]-1)
        #collect minotaur possible actions and store them in m_possibleActions.
        m_possibleActions = []
        if border_of_maze:
            for m_action in list(self.actions.keys())[1:]:
                row_m = self.states[state][2] + self.actions[m_action][0]
                col_m = self.states[state][3] + self.actions[m_action][1]
                outside_of_maze =  (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                    (col_m == -1) or (col_m == self.maze.shape[1])
                if not outside_of_maze:
                    m_possibleActions.append(m_action)
        else:
            m_possibleActions = list(self.actions.keys())[1:]
        return m_possibleActions
    def __move(self, state, action, m_action = None, p_poison=None):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        if p_poison != None:
            if np.random.rand() < p_poison:
                return self.n_states

        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        if row == col == -1:
            return state
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row,col] == 1);
        if m_action != None:
            row_m = self.states[state][2] + self.actions[m_action][0];
            col_m = self.states[state][3] + self.actions[m_action][1];
        else:
            ### Get random action from possible minotaur actions.
            # Get possible minotaur actions.
            m_possibleActions = self.getMinotaur_actions(state)

            # Compute the future position of the minotaur given current state
            minotaur_action = np.random.choice(m_possibleActions)
            row_m = self.states[state][2] + self.actions[minotaur_action][0];
            col_m = self.states[state][3] + self.actions[minotaur_action][1];

        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return self.map[(self.states[state][0], self.states[state][1], row_m, col_m)];
        else:
            return self.map[(row, col, row_m, col_m)];


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            #current minotaur position
            m_possibleActions = self.getMinotaur_actions(s)

            for m_action in m_possibleActions:
                for a in range(self.n_actions):
                    next_s = self.__move(s,a, m_action);
                    transition_probabilities[next_s, s, a] = 1/len(m_possibleActions);
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a);
                    get_eaten = False
                    m_possibleActions = self.getMinotaur_actions(s)
                    # Check if next player position could get eaten
                    for m_action in m_possibleActions:
                        next_Ms = self.__move(s, a, m_action)
                        if self.states[next_Ms][0:2] == self.states[next_Ms][2:]:
                            get_eaten = True

                    # Rewrd for hitting a wall
                    if self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for being eaten
                    elif get_eaten and self.maze[self.states[s][:2]] != 2:
                        rewards[s,a] = self.EATEN_REWARD/len(m_possibleActions)
                    # Reward for reaching the exit
                    elif self.states[s][:2] == self.states[next_s][:2] and self.maze[self.states[next_s][:2]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            self.states[self.n_states] = (-1, -1, -1, -1)
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s], p_poison=P_poison);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while t < 200 and self.maze[self.states[s][0], self.states[s][1]]!=2 and next_s < self.n_states:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s], p_poison=P_poison);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])

                # Update time and state for next iteration
                t +=1
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    states = np.array(list(env.map.keys())+[(-1, -1, -1, -1)])
    poison_p = P_poison
    n_states = states.shape[0]
    n_actions = env.n_actions
    p = np.zeros((n_states, n_states, n_actions))
    # terminal_states = []
    # for state in states[:-1]:
    #     if env.maze[state[0], state[1]] == 2:
    #         terminal_states.append(state)
    # terminal_states = [env.map[tuple(s.tolist())] for s in terminal_states]

    p[:-1, :-1] = env.transition_probabilities * (1-poison_p)
    p[-1,:, :] = poison_p

    r = np.ones([n_states, n_actions]) * -1
    r[:-1, :] = env.rewards
    r[-1, :] = -100 #Death reward
    for state in range(n_states):
        if state == n_states-1:
            r[state, :] = -100 # we die
            p[:, state, :] = 0 #end of the game
            continue
        i, j, m, n = env.states[state]
        if env.maze[i][j] == 2:
            r[state, :] = 0 # finishing
            p[:, state, :] = 0 # end of the game
        elif i == m and j == n: 
            r[state, :] = -100 # we are eaten
            p[:, state, :] = 0 # end of the game
   
    
    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 20:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        # print(n, np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;
  

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):

        if path[i] == (-1, -1, -1, -1):
            grid.get_celld()[(path[i-1][:2])].get_text().set_text('Player')
            grid.get_celld()[(path[i-1][:2])].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[i-1][2:])].set_facecolor(col_map[maze[path[i-1][2:]]])
            grid.get_celld()[(path[i-1][:2])].get_text().set_text('Player is dead')
            break
        grid.get_celld()[(path[i][:2])].get_text().set_text('Player')
        if i > 0:
            if path[i][:2] == path[i-1][:2]:
                grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i-1][2:])].set_facecolor(col_map[maze[path[i-1][2:]]])

                grid.get_celld()[(path[i][:2])].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(path[i-1][2:])].set_facecolor(col_map[maze[path[i-1][2:]]])
                grid.get_celld()[(path[i-1][:2])].set_facecolor(col_map[maze[path[i-1][:2]]])

                grid.get_celld()[(path[i-1][:2])].get_text().set_text('')
        grid.get_celld()[(path[i][:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_PURPLE)
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


