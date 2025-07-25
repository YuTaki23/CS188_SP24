# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    closed = set()
    fringe = util.Stack()
    fringe.push((problem.getStartState(), []))
    while True:
        if fringe.isEmpty():
            return None
        state, action = fringe.pop()
        if problem.isGoalState(state):
            return action
        if state not in closed:
            closed.add(state)
            for successor in problem.getSuccessors(state):
                fringe.push((successor[0], action + [successor[1]]))

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    closed = set()
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))
    while True:
        if fringe.isEmpty():
            return None
        state, action = fringe.pop()
        if problem.isGoalState(state):
            return action
        if state not in closed:
            closed.add(state)
            for successor in problem.getSuccessors(state):
                fringe.push((successor[0], action + [successor[1]]))

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0),
                0)
    while True:
        if fringe.isEmpty():
            return None
        state, action, priority = fringe.pop()
        if problem.isGoalState(state):
            return action
        if state not in closed:
            closed.add(state)
            for successor in problem.getSuccessors(state):
                fringe.update((successor[0], action + [successor[1]], priority + successor[2]),
                              (priority + successor[2]))

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    """
    closed = set()
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0),
                (heuristic(problem.getStartState(), problem)))
    while True:
        if fringe.isEmpty():
            return None
        state, action, priority = fringe.pop()
        if problem.isGoalState(state):
            return action
        if state not in closed:
            closed.add(state)
            for successor in problem.getSuccessors(state):
                fringe.update((successor[0], action + [successor[1]], priority + successor[2]),
                              (priority + successor[2] + heuristic(successor[0], problem)))
"""
    best_g = {}  # 记录到达每个状态的最小代价
    fringe = util.PriorityQueue()
    start = problem.getStartState()
    best_g[start] = 0

    # 存储节点: (状态, 动作序列, 实际代价g)
    fringe.push(
        (start, [], 0),
        heuristic(start, problem)  # 初始优先级 = h(start)
    )

    best_goal = None  # 存储最优目标节点
    best_cost = float('inf')  # 存储到达目标的最小代价

    while not fringe.isEmpty():
        state, actions, g = fringe.pop()
        f = g + heuristic(state, problem)  # 计算f值

        # 如果当前f值已超过已知最优解，提前返回
        if best_goal is not None and f >= best_cost:
            return best_goal

        # 如果当前路径不是最优，跳过
        if g > best_g.get(state, float('inf')):
            continue

        # 如果找到目标且代价更优
        if problem.isGoalState(state):
            if g < best_cost:
                best_goal = actions
                best_cost = g
            continue  # 继续搜索可能更优解

        # 扩展当前节点
        for next_state, action, step_cost in problem.getSuccessors(state):
            new_g = g + step_cost
            # 如果找到更优路径
            if next_state not in best_g or new_g < best_g[next_state]:
                best_g[next_state] = new_g
                new_actions = actions + [action]
                new_f = new_g + heuristic(next_state, problem)
                fringe.push(
                    (next_state, new_actions, new_g),
                    new_f
                )

    return best_goal  # 返回找到的最优解

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
