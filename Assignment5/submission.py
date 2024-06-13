from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    # minimax: return score, action
    def minimax(state, depth, agentIndex):
        # 게임이 끝났거나, 가능한 행동이 없는 경우
        if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
          return (state.getScore(), Directions.STOP)
      
        elif depth == 0:  #if depth == 0 -> eval(s)
          return (self.evaluationFunction(state), Directions.STOP)
        
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth

        results = []
        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
                score = self.evaluationFunction(nextState)
            else:
                score, _ = minimax(nextState, nextDepth, nextAgentIndex)
            results.append((score, action))
        
        if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
            return max(results)
        else:  # 유령일 경우, 최소 점수를 가진 행동을 선택
            return min(results)

    _, action = minimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def minimax(state, depth, agentIndex):
        # 게임이 끝났거나, 가능한 행동이 없는 경우
        if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
          return (state.getScore(), Directions.STOP)
      
        elif depth == 0:  #if depth == 0 -> eval(s)
          return (self.evaluationFunction(state), Directions.STOP)
        
        nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth - 1 if nextAgentIndex == 0 else depth

        results = []
        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
                score = self.evaluationFunction(nextState)
            else:
                score, _ = minimax(nextState, nextDepth, nextAgentIndex)
            results.append((score, action))
        
        if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
            return max(results)
        else:  # 유령일 경우, 최소 점수를 가진 행동을 선택
            return min(results)
    
    # 현재 상태에서 선택한 행동을 취한 다음 상태를 생성
    nextState = gameState.generateSuccessor(0, action)  # 팩맨은 항상 0번 에이전트.
    
    # 다음 상태에 대해 minimax 값을 계산. 팩맨의 행동 후 첫 번째 유령부터 시작하므로 agentIndex는 1.
    score, _ = minimax(nextState, self.depth, 1)  # 팩맨이 행동한 후 첫 번째 유령(1번 에이전트)의 차례
    # 선택한 행동에 대한 Q-value(여기서는 minimax 값)를 반환
    return score

    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def expectimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      for action in state.getLegalActions(agentIndex):
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = expectimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:  # 유령일 경우, 최소 점수를 가진 행동을 선택
        # 기댓값
        ExpectedScore = sum(score for score, action in results) / len(results)
        # 평균 점수와 임의의 행동을 반환 (유령의 특정 행동 선택은 여기서는 딱히 중요하지 않음)
        return (ExpectedScore, None)

    _, action = expectimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def expectimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      for action in state.getLegalActions(agentIndex):
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = expectimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:  # 유령일 경우, 최소 점수를 가진 행동을 선택
        # 기댓값
        ExpectedScore = sum(score for score, action in results) / len(results)
        # 평균 점수와 임의의 행동을 반환 (유령의 특정 행동 선택은 여기서는 딱히 중요하지 않음)
        return (ExpectedScore, None)
      
    nextState = gameState.generateSuccessor(0, action)

    score, _ = expectimax(nextState, self.depth, 1)  # 팩맨이 행동한 후 첫 번째 유령(1번 에이전트)의 차례
    
    # 선택한 행동에 대한 Q-value(여기서는 expectimax 값)를 반환
    return score
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def biasedexpectimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      actions = state.getLegalActions(agentIndex)
      for action in actions:
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = biasedexpectimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:  # 유령일 경우
        p_stop = 0.5 + 0.5 * 1/len(actions)
        p_else = 0.5 * 1/len(actions)

        # 기댓값
        ExpectedScore = sum(score * (p_stop if action == Directions.STOP else p_else) for score, action in results)
        # 기댓값을 기반으로 행동 선택
        actionProbabilities = [p_stop if action == Directions.STOP else p_else for _, action in results]
        chosen_action = random.choices(actions, weights=actionProbabilities)
        
        #weighted_actions = [(action, p_stop if action == Directions.STOP else p_else) for _, action in results]
        #_, chosen_action = max(weighted_actions, key=lambda x: x[1])  # 확률이 높은 행동 선택
        return (ExpectedScore, chosen_action)

    _, action = biasedexpectimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def biasedexpectimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      actions = state.getLegalActions(agentIndex)
      for action in actions:
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = biasedexpectimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:  # 유령일 경우, 최소 점수를 가진 행동을 선택
        p_stop = 0.5 + 0.5 * 1/len(actions)
        p_else = 0.5 * 1/len(actions)

        # 기댓값
        ExpectedScore = sum(score * (p_stop if action == Directions.STOP else p_else) for score, action in results)
        # 기댓값을 기반으로 행동 선택
        actionProbabilities = [p_stop if action == Directions.STOP else p_else for _, action in results]
        chosen_action = random.choices(actions, weights=actionProbabilities)
        
        #weighted_actions = [(action, p_stop if action == Directions.STOP else p_else) for _, action in results]
        #_, chosen_action = max(weighted_actions, key=lambda x: x[1])  # 확률이 높은 행동 선택
        return (ExpectedScore, chosen_action)
      
    nextState = gameState.generateSuccessor(0, action)
    score, _ = biasedexpectimax(nextState, self.depth, 1)  # 팩맨이 행동한 후 첫 번째 유령(1번 에이전트)의 차례
    
    return score
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def expectiminimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      for action in state.getLegalActions(agentIndex):
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = expectiminimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:
        if agentIndex % 2 == 0:  # 짝수 번호의 유령은 동일한 확률로 행동을 선택 (기댓값 계산)
          expectedValue = sum(result[0] for result in results) / len(results)
          return (expectedValue, None)
        else:  # 홀수 번호의 유령은 최소 점수를 가진 행동을 선택
          return min(results)
        
    _, action = expectiminimax(gameState, self.depth, 0)
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def expectiminimax(state, depth, agentIndex):
      # 게임이 끝났거나, 가능한 행동이 없는 경우
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
        
      nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth

      results = []
      for action in state.getLegalActions(agentIndex):
        nextState = state.generateSuccessor(agentIndex, action)
        if nextDepth == 0 or nextState.isWin() or nextState.isLose():  # 최대 깊이 도달 또는 게임 종료 상태
          score = self.evaluationFunction(nextState)
        else:
          score, _ = expectiminimax(nextState, nextDepth, nextAgentIndex)
        results.append((score, action))
        
      if agentIndex == 0:  # 팩맨일 경우, 최대 점수를 가진 행동을 선택
        return max(results)
      else:
        if agentIndex % 2 == 0:  # 짝수 번호의 유령은 동일한 확률로 행동을 선택 (기댓값 계산)
          expectedValue = sum(result[0] for result in results) / len(results)
          return (expectedValue, None)
        else:  # 홀수 번호의 유령은 최소 점수를 가진 행동을 선택
          return min(results)
        
    nextState = gameState.generateSuccessor(0, action)
    score, _ = expectiminimax(nextState, self.depth, 1)  # 팩맨이 행동한 후 첫 번째 유령(1번 에이전트)의 차례
  
    return score
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def alphaBeta(state, depth, agentIndex, alpha, beta):
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
    
      numAgents = state.getNumAgents()
      nextAgentIndex = (agentIndex + 1) % numAgents
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth
    
      if agentIndex == 0:  # Max agent (Pacman)
        value = float("-inf")
        bestAction = Directions.STOP
        for action in state.getLegalActions(agentIndex):
          nextState = state.generateSuccessor(agentIndex, action)
          nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
          if nextValue > value: # value가 더 크다면, value랑 action update
            value, bestAction = nextValue, action
          alpha = max(alpha, value)
          if alpha >= beta:  # Prune
            break
        return value, bestAction
      else:  # Min agent (Ghosts)
        actions = state.getLegalActions(agentIndex)
        # Odd-numbered Ghosts, minimize
        if agentIndex % 2 == 1:
            value = float("inf")
            for action in actions:
                nextState = state.generateSuccessor(agentIndex, action)
                nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
                value = min(value, nextValue)
                beta = min(beta, value)
                if beta <= alpha:  # Prune
                    break
            return value, Directions.STOP
        # Even-numbered Ghosts, expectimax
        else:
            totalValue = 0
            for action in actions:
                nextState = state.generateSuccessor(agentIndex, action)
                nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
                totalValue += nextValue
            avgValue = totalValue / len(actions) if actions else 0
            return avgValue, Directions.STOP

    _, action = alphaBeta(gameState, self.depth, 0, float("-inf"), float("inf"))
    return action
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    #raise NotImplementedError  # remove this line before writing code
    def alphaBeta(state, depth, agentIndex, alpha, beta):
      if state.isWin() or state.isLose() or state.getLegalActions(agentIndex) == [Directions.STOP]:
        return (state.getScore(), Directions.STOP)
      
      elif depth == 0:  #if depth == 0 -> eval(s)
        return (self.evaluationFunction(state), Directions.STOP)
    
      numAgents = state.getNumAgents()
      nextAgentIndex = (agentIndex + 1) % numAgents
      nextDepth = depth - 1 if nextAgentIndex == 0 else depth
    
      if agentIndex == 0:  # Max agent (Pacman)
        value = float("-inf")
        bestAction = Directions.STOP
        for action in state.getLegalActions(agentIndex):
          nextState = state.generateSuccessor(agentIndex, action)
          nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
          if nextValue > value:
            value, bestAction = nextValue, action
          alpha = max(alpha, value)
          if alpha >= beta:  # Prune
            break
        return value, bestAction
      else:  # Min agent (Ghosts)
        actions = state.getLegalActions(agentIndex)
        # Odd-numbered Ghosts, minimize
        if agentIndex % 2 == 1:
            value = float("inf")
            for action in actions:
                nextState = state.generateSuccessor(agentIndex, action)
                nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
                value = min(value, nextValue)
                beta = min(beta, value)
                if beta <= alpha:  # Prune
                    break
            return value, Directions.STOP
        # Even-numbered Ghosts, expectimax
        else:
            totalValue = 0
            for action in actions:
                nextState = state.generateSuccessor(agentIndex, action)
                nextValue, _ = alphaBeta(nextState, nextDepth, nextAgentIndex, alpha, beta)
                totalValue += nextValue
            avgValue = totalValue / len(actions) if actions else 0
            return avgValue, Directions.STOP

    nextState = gameState.generateSuccessor(0, action)
    # 다음 상태에서의 최적 점수 계산
    score, _ = alphaBeta(nextState, self.depth, 1, float("-inf"), float("inf"))
    
    return score
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  #raise NotImplementedError  # remove this line before writing code
  def getNearestDistance(objectsPositions, currentPosition):
    return min([manhattanDistance(currentPosition, pos) for pos in objectsPositions]) if objectsPositions else float("inf")
    
  def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

  pacmanPosition = currentGameState.getPacmanPosition()
  foodPositions = currentGameState.getFood().asList()
  capsulePositions = currentGameState.getCapsules()
  ghostsStates = [currentGameState.getGhostState(i) for i in range(1, currentGameState.getNumAgents())]
  scaredGhosts = [ghost for ghost in ghostsStates if ghost.scaredTimer > 0]
  activeGhosts = [ghost for ghost in ghostsStates if ghost.scaredTimer == 0]
  
  nearestFoodDist = getNearestDistance(foodPositions, pacmanPosition)
  nearestCapsuleDist = getNearestDistance(capsulePositions, pacmanPosition)
  nearestScaredGhostDist = getNearestDistance([ghost.getPosition() for ghost in scaredGhosts], pacmanPosition)
  nearestActiveGhostDist = getNearestDistance([ghost.getPosition() for ghost in activeGhosts], pacmanPosition)
  
  numScaredGhosts = len(scaredGhosts)
  scoreFromCurrentState = currentGameState.getScore()
  
  if numScaredGhosts > 0:
    nearestDist = nearestScaredGhostDist
    scoreForCapsules = 55 * len(capsulePositions)
  else:
    nearestDist = nearestCapsuleDist if capsulePositions else nearestFoodDist
    scoreForCapsules = 0

  # Adjusted weights, last one is for active ghosts
  weights = [1.0, 10.13, 150.0, 1.0, -1.2]  
  features = [scoreFromCurrentState, 1.0 / nearestDist, 1.0 / (len(capsulePositions) + 1), scoreForCapsules, 1.0 / (nearestActiveGhostDist + 1)]

  evalScore = sum(w * f for w, f in zip(weights, features))

  return evalScore
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  #raise NotImplementedError  # remove this line before writing code
  return 'AlphaBetaAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
