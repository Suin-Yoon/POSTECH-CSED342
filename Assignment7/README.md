# CSED342 Assignment 7. Car Tracking 요약 및 방향성

지도에 검정 차(우리 거)랑 회색 차 여러 개가 있음. 회색 차의 실제 위치(C_t)는 알려져 있지 않고, 현재 검정 차의 위치에서 회색 차까지의 거리 정보(D_t)만 알려져 있음.

해야 할 것: 사후 예측 분포(Posterior distribution) P(C_t | D_1 = d_1, ..., D_t = d_t)를 대략적으로 계산하는 car tracker 구현.  
즉, 각 회색 차들까지의 거리를 바탕으로 회색 차들의 실제 위치를 대략적으로 계산하는 것을 목표로 함.  

Python drive.py -d로 차 위치 확인 가능(debug)  
문제1을 해결해야 맵에서 색깔 표시 가능  
문제2 해결하면 색깔타일들이 움직일 것  
유의사항 꼭 참고! 옳은 방식으로 구현하기  


개념

2번  
* transition probability: 한 state에서 다른 state르 전이될 확률
* current probability: 현재 시점에서 각 타일에 차가 위치할 확률
* emission probability: transition probability와 current probability의 곱. 새로운 타일로 전이될 확률.
* 마지막으로, belief 상태를 정규화하여 확률의 합이 1이 되도록 함.
 

3번 https://velog.io/@soup1997/Particle-Filter 참고  
Particle filtering: 시간에 따라 변하는 시스템의 상태를 추정. 비선형 또는 비가우시안 noise를 가진 시스템의 상태 추정에 유용.
0) 초기화: 시스템의 초기 상태를 나타내는 파티클들(또는 샘플들)을 생성, 각 파티클은 시스템의 가능한 상태 중 하나를 나타냄.
1) 파티클 예측: 시간이 경과함에 따라 각 파티클의 상태를 시스템의 모델을 사용하여 계산
2) 파티클 가중치 업데이트: 측정값과 각 파티클 예측 추정값의 차이를 기반으로 가중치 갱신. 그 후 가중치를 정규화
3) 추정값 계산: 각 파티클의 가중치를 고려한 가중합으로 추정값 계산
4) Resampling: 각 파티클의 가중치를 반영해 파티클을 새로 생성
