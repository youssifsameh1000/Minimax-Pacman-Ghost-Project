# ghostAgents.py
# --------------
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()
        return dist


# ---------------------------------------------------------------------------
# Minimax Ghost Agents  (CS188 Project – Recursive vs. Iterative)
# ---------------------------------------------------------------------------

def _ghostEvaluation(state, ghostIndex):
    """
    Default evaluation from the ghost's perspective.
    A lower Pacman score is better for the ghost, so we negate it.
    We also reward the ghost for being CLOSER to Pacman (smaller distance = higher eval).
    """
    score = -state.getScore()

    # Terminal bonuses
    if state.isLose():   # Pacman lost → ghost wins
        score += 500
    if state.isWin():    # Pacman won  → ghost loses
        score -= 500

    # Distance heuristic: reward being close to Pacman
    pacmanPos = state.getPacmanPosition()
    ghostPos = state.getGhostPosition(ghostIndex)
    distance = manhattanDistance(pacmanPos, ghostPos)
    # Subtract distance so closer = higher score for the ghost
    score -= 2 * distance

    return score


class MinimaxGhostRecursive(GhostAgent):
    """
    Version A – Recursive minimax ghost with alpha-beta pruning.
    This ghost is the MAX player at its own turn; all other agents are MIN.
    Depth counts one full round of (all agents move once) as 1 ply.
    """

    def __init__(self, index, depth=2, evalFn=None):
        super().__init__(index)
        self.depth = depth
        self.evalFn = evalFn or _ghostEvaluation
        # Instrumentation (used by benchmark)
        self._maxRecursionDepth = 0

    def getDistribution(self, state):
        self._maxRecursionDepth = 0
        _, bestAction = self._minimax(state, self.depth, self.index, float('-inf'), float('inf'), 0)
        dist = util.Counter()
        if bestAction is not None:
            dist[bestAction] = 1.0
        else:
            # Fallback: shouldn't happen, but pick first legal action
            actions = state.getLegalActions(self.index)
            if actions:
                dist[actions[0]] = 1.0
        return dist

    def _minimax(self, state, depth, agentIndex, alpha, beta, recursionLevel):
        """
        Returns (value, bestAction).
        *agentIndex* cycles through all agents. When it wraps back to
        self.index, we decrement depth.
        """
        self._maxRecursionDepth = max(self._maxRecursionDepth, recursionLevel)

        numAgents = state.getNumAgents()

        # Terminal test
        if state.isWin() or state.isLose() or depth == 0:
            return self.evalFn(state, self.index), None

        legalActions = state.getLegalActions(agentIndex)
        if not legalActions:
            return self.evalFn(state, self.index), None

        # Determine next agent & depth
        nextAgent = (agentIndex + 1) % numAgents
        nextDepth = depth - 1 if nextAgent == self.index else depth

        isMax = (agentIndex == self.index)

        if isMax:
            # MAX node (this ghost's turn)
            bestValue = float('-inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                val, _ = self._minimax(successor, nextDepth, nextAgent, alpha, beta, recursionLevel + 1)
                if val > bestValue:
                    bestValue = val
                    bestAction = action
                alpha = max(alpha, bestValue)
                if bestValue > beta:
                    break  # beta cut-off
            return bestValue, bestAction
        else:
            # MIN node (any other agent)
            bestValue = float('inf')
            bestAction = None
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                val, _ = self._minimax(successor, nextDepth, nextAgent, alpha, beta, recursionLevel + 1)
                if val < bestValue:
                    bestValue = val
                    bestAction = action
                beta = min(beta, bestValue)
                if bestValue < alpha:
                    break  # alpha cut-off
            return bestValue, bestAction


class MinimaxGhostIterative(GhostAgent):
    """
    Version B – Iterative minimax ghost using explicit stack frames.
    Produces the exact same moves as MinimaxGhostRecursive.
    """

    def __init__(self, index, depth=2, evalFn=None):
        super().__init__(index)
        self.depth = depth
        self.evalFn = evalFn or _ghostEvaluation
        # Instrumentation
        self._maxStackSize = 0

    def getDistribution(self, state):
        self._maxStackSize = 0
        _, bestAction = self._minimax_iterative(state, self.depth)
        dist = util.Counter()
        if bestAction is not None:
            dist[bestAction] = 1.0
        else:
            actions = state.getLegalActions(self.index)
            if actions:
                dist[actions[0]] = 1.0
        return dist

    def _minimax_iterative(self, state, depth):
        """
        Iterative minimax with alpha-beta pruning using an explicit stack.
        Each stack frame mirrors one call of the recursive version.
        """
        numAgents = state.getNumAgents()

        # Stack frame fields:
        #   state, depth, agentIndex, alpha, beta,
        #   isMax, actions, actionIdx, bestValue, bestAction, phase
        # phase: 'expand' = need to push child, 'collect' = child returned a value
        EXPAND = 0
        COLLECT = 1

        root = {
            'state': state,
            'depth': depth,
            'agentIndex': self.index,
            'alpha': float('-inf'),
            'beta': float('inf'),
            'isMax': True,
            'actions': None,
            'actionIdx': 0,
            'bestValue': None,
            'bestAction': None,
            'phase': EXPAND,
        }
        stack = [root]
        returnValue = None   # value passed back from child to parent

        while stack:
            self._maxStackSize = max(self._maxStackSize, len(stack))
            frame = stack[-1]

            if frame['phase'] == EXPAND:
                s = frame['state']
                d = frame['depth']
                ai = frame['agentIndex']

                # Terminal check
                if s.isWin() or s.isLose() or d == 0:
                    returnValue = (self.evalFn(s, self.index), None)
                    stack.pop()
                    continue

                actions = s.getLegalActions(ai)
                if not actions:
                    returnValue = (self.evalFn(s, self.index), None)
                    stack.pop()
                    continue

                frame['actions'] = actions
                frame['isMax'] = (ai == self.index)
                frame['bestValue'] = float('-inf') if frame['isMax'] else float('inf')
                frame['bestAction'] = None
                frame['actionIdx'] = 0
                frame['phase'] = COLLECT

                # Push first child
                action = actions[0]
                successor = s.generateSuccessor(ai, action)
                nextAgent = (ai + 1) % numAgents
                nextDepth = d - 1 if nextAgent == self.index else d

                child = {
                    'state': successor,
                    'depth': nextDepth,
                    'agentIndex': nextAgent,
                    'alpha': frame['alpha'],
                    'beta': frame['beta'],
                    'isMax': False,
                    'actions': None,
                    'actionIdx': 0,
                    'bestValue': None,
                    'bestAction': None,
                    'phase': EXPAND,
                }
                stack.append(child)

            elif frame['phase'] == COLLECT:
                # We just got a return value from a child
                childVal = returnValue[0]
                action = frame['actions'][frame['actionIdx']]

                if frame['isMax']:
                    if childVal > frame['bestValue']:
                        frame['bestValue'] = childVal
                        frame['bestAction'] = action
                    frame['alpha'] = max(frame['alpha'], frame['bestValue'])
                    pruned = frame['bestValue'] > frame['beta']
                else:
                    if childVal < frame['bestValue']:
                        frame['bestValue'] = childVal
                        frame['bestAction'] = action
                    frame['beta'] = min(frame['beta'], frame['bestValue'])
                    pruned = frame['bestValue'] < frame['alpha']

                frame['actionIdx'] += 1

                # Check if we still have actions left and no pruning
                if not pruned and frame['actionIdx'] < len(frame['actions']):
                    # Push next child
                    ai = frame['agentIndex']
                    nextAction = frame['actions'][frame['actionIdx']]
                    successor = frame['state'].generateSuccessor(ai, nextAction)
                    nextAgent = (ai + 1) % numAgents
                    nextDepth = frame['depth'] - 1 if nextAgent == self.index else frame['depth']

                    child = {
                        'state': successor,
                        'depth': nextDepth,
                        'agentIndex': nextAgent,
                        'alpha': frame['alpha'],
                        'beta': frame['beta'],
                        'isMax': False,
                        'actions': None,
                        'actionIdx': 0,
                        'bestValue': None,
                        'bestAction': None,
                        'phase': EXPAND,
                    }
                    stack.append(child)
                else:
                    # Done with this frame – return value to parent
                    returnValue = (frame['bestValue'], frame['bestAction'])
                    stack.pop()

        return returnValue
