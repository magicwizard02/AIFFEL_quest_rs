# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최에리나
- 리뷰어 : 정주열

# PRT(Peer Review Template)
- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 한국어 NLP 전처리부터 모델 학습, 평가, 임베딩까지 전체 파이프라인이 잘 구현되었음.
    <img width="813" height="627" alt="image" src="https://github.com/user-attachments/assets/d5e2a141-ce61-4c4c-9f9f-6681cc245ad0" />
<img width="1443" height="394" alt="image" src="https://github.com/user-attachments/assets/12125a99-02bb-4cb4-abb7-e7192c25bf13" />

- [x] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 자연어 처리 모델의 입력층을 구성하기 위해 사전을 구축하고 인코딩하는 필수 단계이다. build_vocabulary 함수와 encode_sentences 함수 상단에 기능 설명이 적절히 작성되어 있으며, 코드 내부에도 1번부터 5번까지 단계별로 주석이 달려 있어 흐름을 파악하기 매우 용이했다.
    <img width="549" height="661" alt="image" src="https://github.com/user-attachments/assets/e0e845cb-c42d-4500-89ac-287526bfe8cb" />
    <img width="669" height="374" alt="image" src="https://github.com/user-attachments/assets/508da6c3-aee6-4d72-9e5c-f195acc4309f" />

- [x] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 단순히 모델을 학습시키는 데 그치지 않고, 다양한 실험 조건을 체계적으로 분석한 기록함.
    <img width="611" height="184" alt="image" src="https://github.com/user-attachments/assets/2cf3d0bb-d803-4430-9aa4-51e6db20bfef" />

- [x] **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 단순한 코드 작성을 넘어, 프로젝트 전반에 대한 깊이 있는 통찰이 담긴 회고가 작성되었음.
    <img width="1388" height="650" alt="image" src="https://github.com/user-attachments/assets/601ca77c-cae3-482f-96a8-9b0ba1940e56" />

- [x] **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 전체적으로 코드가 매우 구조화되어 있으며, 간결하고 효율적으로 작성되었음.
    <img width="630" height="399" alt="image" src="https://github.com/user-attachments/assets/48440bf7-d7fb-4d30-a944-654a73f8e797" />

# 회고(참고 링크 및 코드 개선)
단순히 모델을 돌려보는 것에 그치지 않고, Scratch Training과 Word2Vec Fine-Tuning의 차이점을 Summary Table로 정리한 부분이 매우 인상적이었습니다. 이를 통해 전이 학습의 효율성을 한눈에 파악할 수 있었습니다. build_vocabulary와 encode_sentences 함수를 분리하여 구현한 덕분에 전처리 파이프라인이 매우 깔끔합니다. 특히 특수 토큰(<PAD>, <BOS>, <UNK>)을 예약 인덱스로 관리하는 방식은 실제 서비스 환경에서도 활용하기 좋은 구조라고 생각합니다.
