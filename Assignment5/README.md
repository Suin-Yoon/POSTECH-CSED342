# CSED342 Assignment 5. Multi-agent Pac-Man 요약 및 방향성
max agent 1개: 팩맨
min agent 여러 개(multi min agents): 유령들 -> 유령 개수에 상관없이 동작해야 함.
각 max layer(팩맨)마다 여러 min layers(유령들)가 있도록 game tree를 확장해야 함. 이 game tree는 self.depth, self.evaluationFunction을 참고하여 임의의 depth까지 확장되어야 함.

1. minimax
* 평가함수 -> 이미 제공된 self.evaluationFunction 사용. 이는 게임 state를 평가하여 점수를 반환하고, minimax 알고리즘에서 leaf node 평가 시 사용된다.
* 초기상태 minimax 값: 깊이 1, 2, 3, 4에 대해 각각 9, 8, 7, -492
* 팩맨: 에이전트 0 / 에이전트는 인덱스가 증가하는 방향?으로 움직임.
* 모든 minimax state는 GameState로 관리됨. getAction에 전달되거나, GameState.generateSuccessor를 통해 생성됨.
* getAction에서는 Vmax, min을 사용해서 최선의 행동 결정
* 여러 행동이 최선의 움직임으로 동일한 가치를 가질 경우, 이러한 무승부를 처리하는 방법을 선택할 수 있음..
* getQ: 현재 상태와 주어진 행동에 대한 Qmax, min 반환

2. expectimax
모든 유령이 균등하게 무작위로 움직인다는 가정 하에, 유령의 행동을 평균 점수(기댓값)로 모델링. 팩맨의 경우에는 여전히 최대 점수를 추구하지만, 유령의 경우에는 가능한 모든 행동의 평균 점수를 계산.

3. biased expectimax
 모든 유령이 그들의 가능한 행동들 중에서 '정지(STOP)' 행동을 선택할 확률을 높게 설정한 biased-expectimax 알고리즘을 구현. 유령의 각 행동 선택 확률은 다음과 같음:

'정지(STOP)' 행동: (p(a in A) = 0.5 + 0.5 * 1/len(A))
그 외의 행동: (p(a in A) = 0.5 * 1/len(A))
여기서 (A)는 유령이 선택할 수 있는 행동들의 집합( = actions)

4. expectiminimax

minimax와 달리, 현실에서는 모든 상대가 같은 전략을 따르지 않는다. 이 문제에서는 상대방(유령)을 두 그룹으로 나누어, 한 그룹은 Min 정책을 따르고 다른 그룹은 무작위 정책을 따른다고 가정. 특히 홀수 번호를 가진 유령은 Pac-Man에게 최악의 값을 선택하는 Min 정책을 따르고, 짝수 번호의 유령은 일정한 분포(이 경우에는 균등 분포)에 따라 무작위로 행동을 선택

홀수 번호의 유령에 대해서는 최소 값을 선택하는 Min 단계를, 짝수 번호의 유령에 대해서는 가능한 모든 행동의 기댓값을 계산하는 Expectation 단계를 사용
일부 유령이 Pac-Man에게 최악의 행동을 선택하고, 다른 일부는 균등하게 무작위로 행동을 선택하는 상황에서 Pac-Man의 행동을 결정하는 것이 목표

* 문제에서 언급한 Expectiminimax와 Minimax 값이 깊이 1, 2, 3에서는 같고 깊이 4에서만 다른 이유는, 낮은 깊이에서는 Expectation 단계의 영향이 크지 않지만, 깊이가 깊어질수록 무작위로 행동을 선택하는 유령의 기댓값 계산이 결과에 영향을 미치기 시작하기 때문. 깊이 4에서는 무작위로 행동을 선택하는 유령의 영향이 결과값에 반영되어 Expectiminimax와 Minimax 값이 차이나게 된다.


5. alpha-beta pruning
alpha-beta pruning이란: 불필요한 노드의 탐색을 줄여 탐색 속도를 향상시키는 기술(pruning: 가지치기)
이 문제에서는 기본적인 알파-베타 프루닝 알고리즘을 여러 개의 expectiminimizer 에이전트들에게 적용하는 것을 포함하여, 약간 더 일반화된 형태로 확장해야 함.

alpha-beta pruning 기본 개념
alpha-beta pruning은 미니맥스 알고리즘의 효율을 개선하기 위한 기법. 각 노드에서 최적의 수를 찾기 위해 모든 가능한 경우의 수를 탐색하는 대신, alpha(최솟값의 최대한의 하한)와 beta(최댓값의 최소한의 상한)를 이용하여 탐색 공간을 줄인다. 즉, 탐색 중에 alpha 값이 beta 값보다 크거나 같은 경우, 그 branch는 최적의 결과를 가져올 수 없다고 판단하여 더 이상 탐색하지 않는다(pruning).

구현요소:
alpha는 -inf, beta는 +inf로 초기화.
재귀적 탐색과 pruning: expectiminimax 탐색을 진행하면서, 가능한 경우의 수를 평가하고, alpha와 beta 값을 업데이트. 탐색 과정에서 alpha가 beta보다 크거나 같은 경우, 해당 branch의 탐색을 중단.

