import util, math, random
from collections import defaultdict
from util import ValueIteration
import numpy as np


############################################################
# Problem 1a: Volcano Crossing


class VolcanoCrossing():
    """
    grid_world: a 2D numpy array where 0 is explorable, negative integer is a volcano, and positive integer is the goal.
    discount: discount factor
    moveReward: reward of moving from one cell to another
    value_table: a 2D numpy array where each cell represents the value of the cell
    actions: a list of possible actions
    """
    def __init__(self, grid_world, discount=1, moveReward=-1):
        self.grid_world = grid_world
        self.discount = discount
        self.moveReward = moveReward
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Return the value table after running |numIters| of value iteration.
    # You do not need to modify this function.
    def value_iteration(self, numIters=1):
        self.value_table = np.zeros(self.grid_world.shape) # Initialize value table

        for _ in range(numIters):
            self.value_table = self.value_update(self.value_table)
        return self.value_table

    # Return the state is Volcano or Island.
    # You do not need to modify this function.
    # If the state is Volcano or Island, return True.
    # Otherwise(self.grid_world[state] == 0), return False.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def is_volcano_or_island(self, state):
        return self.grid_world[state] != 0

    # Checks if the agent can move to the next state.
    # This function below has been implemented for your convenience, but it is not necessarily required to be used.
    def movable(self, state, action):
        x, y = state
        i, j = action
        return 0 <= x + i < self.grid_world.shape[0] and 0 <= y + j < self.grid_world.shape[1]


    # Return the value table after updating the value of each grid cell.
    def value_update(self, value_table):
        # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
        new_value_table = np.copy(value_table)  # 가치 테이블의 복사본을 생성하여 업데이트.
        
        for x in range(self.grid_world.shape[0]):
            for y in range(self.grid_world.shape[1]):
                state = (x, y)
                if self.is_volcano_or_island(state):  # 화산 또는 섬 셀인 경우, 해당 셀의 가치는 그리드 월드의 해당 셀 값과 동일.
                    new_value_table[state] = self.grid_world[state]
                else:
                    values = []
                    for action in self.actions:
                        if self.movable(state, action):
                            next_state = (x + action[0], y + action[1])
                            # 다음 상태의 가치 계산
                            value = self.moveReward + self.discount * value_table[next_state]   # Reward + Discount
                            values.append(value)
                        else:
                            # 현재 셀의 가치를 사용하여 경계를 넘지 않도록 함.
                            values.append(self.moveReward + self.discount * value_table[state])
                    new_value_table[state] = max(values)  # 가능한 모든 행동에 대해 최대 가치를 선택.
        return new_value_table
        # END_YOUR_ANSWER

############################################################
# Problem 2a: BlackjackMDP


class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        super().__init__()

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_ANSWER (our solution is 44 lines of code, but don't worry if you deviate from this)
        totalCardValueInHand, nextCardIndexIfPeeked, deckCardCounts = state
        results = []

        if deckCardCounts == None:  # 이미 게임이 끝난 상태
            return results
        
        if action == 'Quit':
            if ( totalCardValueInHand > self.threshold ):
                return [((totalCardValueInHand, None, None), 1, 0)]
            else:
                return [((totalCardValueInHand, None, None), 1, totalCardValueInHand)]
            
        
        if action == 'Peek':
            if nextCardIndexIfPeeked != None:  #  if the player peeks twice in a row
                return []
            
            for i, cardCount in enumerate(deckCardCounts):
                if cardCount > 0:
                    prob = cardCount / sum(deckCardCounts)
                    results.append(((totalCardValueInHand, i, deckCardCounts), prob, -self.peekCost))
            return results

        if action == 'Take':
            if nextCardIndexIfPeeked != None:
                # Take the peeked card
                cardValue = self.cardValues[nextCardIndexIfPeeked]
                newTotal = totalCardValueInHand + cardValue
                newDeckCounts = list(deckCardCounts)
                newDeckCounts[nextCardIndexIfPeeked] -= 1
                if newTotal > self.threshold:  # Bust
                    results.append(((newTotal, None, None), 1, 0))
                elif all(count == 0 for count in newDeckCounts): # Run out of cards
                    results.append(((newTotal, None, None), 1, newTotal))
                else: # Continue the game
                    results.append(((newTotal, None, tuple(newDeckCounts)), 1, 0))
            else:
                # last action is not peek -> take card randomly
                totalProb = sum(deckCardCounts) 
                for i, cardValue in enumerate(self.cardValues):
                    if deckCardCounts[i] > 0:
                        newTotal = totalCardValueInHand + cardValue
                        newDeckCounts = list(deckCardCounts)
                        newDeckCounts[i] -= 1
                        prob = deckCardCounts[i] / totalProb
                        
                        if newTotal > self.threshold:  # Bust
                            results.append(((newTotal, None, None), prob, 0))
                        elif all(count == 0 for count in newDeckCounts): # Run out of cards
                            results.append(((newTotal, None, None), prob, newTotal))
                        else: # Continue the game
                            results.append(((newTotal, None, tuple(newDeckCounts)), prob, 0))
            return results

        
        # END_YOUR_ANSWER

    def discount(self):
        return 1


############################################################
# Problem 3a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class Qlearning(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with episode=[..., state, action,
    # reward, newState], which you should use to update
    # |self.weights|. You should update |self.weights| using
    # self.getStepSize(); use self.getQ() to compute the current
    # estimate of the parameters. Also, you should assume that
    # V_opt(newState)=0 when isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        state, action, reward, newState = episode[-4:]

        if isLast(state):
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        if isLast(newState):
            V_opt = 0
        else:
            # V_opt = max(self.getQ(newState, nextAction) for nextAction in self.actions(newState))로 바꿀 수 있을 듯?
            maxQ = float('-inf')
            for nextAction in self.actions(newState):
                qValue = self.getQ(newState, nextAction)
                if qValue > maxQ:
                    maxQ = qValue
            V_opt = maxQ

        # 현재 상태와 행동에 대한 Q 값을 계산.
        hat_Q_opt = self.getQ(state, action)

        eta = self.getStepSize()
        target = reward + self.discount * V_opt
        prediction = hat_Q_opt
        td_error = target - prediction

        for feature, value in self.featureExtractor(state, action):
            self.weights[feature] += eta * td_error * value
        
        # END_YOUR_ANSWER


############################################################
# Problem 3b: Q SARSA

class SARSA(Qlearning):
    # We will call this function with episode=[..., state, action,
    # reward, newState, newAction, newReward, newNewState], which you
    # should use to update |self.weights|. You should
    # update |self.weights| using self.getStepSize(); use self.getQ()
    # to compute the current estimate of the parameters. Also, you
    # should assume that Q_pi(newState, newAction)=0 when when
    # isLast(newState) is True
    def incorporateFeedback(self, episode, isLast):
        assert (len(episode) - 1) % 3 == 0
        if len(episode) >= 7:
            state, action, reward, newState, newAction = episode[-7: -2]
        else:
            return

        # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
        hat_Q_pi = self.getQ(state, action)
        
        # newState가 마지막 상태인 경우, Q_pi(newState, newAction) = 0
        if isLast(newState):
            Q_pi_newState_newAction = 0
        else:
        # newState와 newAction에 대한 Q 값을 계산(SARSA 클래스가 QLearning 클래스를 상속받으므로 getQ 사용가능)
            Q_pi_newState_newAction = self.getQ(newState, newAction)

        eta = self.getStepSize()
        target = reward + self.discount * Q_pi_newState_newAction

        # TD 오차 계산
        td_error = target - hat_Q_pi

        for feature, value in self.featureExtractor(state, action):
            self.weights[feature] += eta * td_error * value
        # END_YOUR_ANSWER

# Return a singleton list containing indicator feature (if exist featurevalue = 1)
# for the (state, action) pair.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 3c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs
# (see identityFeatureExtractor() above for an example).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card type and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card type is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).
#       Example: if the deck is (3, 4, 0, 2), you should have four features (one for each card type).
#       And the first feature key will be (0, 3, action)
#       Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_ANSWER (our solution is 8 lines of code, but don't worry if you deviate from this)
    features = []

    # 첫 번째 feature: 총합과 행동에 대한 지표
    features.append(((total, action), 1))

    if counts != None:  # 카드 덱의 정보가 있는 경우에만 추가 feature을 추출
        # 두 번째 feature: 각 카드 유형의 존재/부재와 행동에 대한 지표
        presence = tuple(int(count > 0) for count in counts)
        features.append(((presence, action), 1))
        
        # 세 번째 feature: 각 카드 유형별 카드 수와 행동에 대한 특징
        for i, count in enumerate(counts):
            features.append((((i, count, counts[i]), action), 1))

    return features
    # END_YOUR_ANSWER