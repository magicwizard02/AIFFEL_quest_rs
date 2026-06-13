# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최에리나
- 리뷰어 : 김도현

# PRT(Peer Review Template)
- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부

모델 평가

<img width="855" height="325" alt="Image" src="https://github.com/user-attachments/assets/b7a8aae2-ebcc-4884-846c-5e8ea2c647d8" />

실제요약과 비교

<img width="803" height="664" alt="Image" src="https://github.com/user-attachments/assets/73ebc0d5-b601-4528-bc2c-ebc60130eb69" />

추출적요약과 추상적요약 비교

<img width="805" height="416" alt="Image" src="https://github.com/user-attachments/assets/a701bb3f-6185-44eb-a0ec-7ae01786c846" />


- [x] **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

<img width="621" height="183" alt="Image" src="https://github.com/user-attachments/assets/729b9554-3ae2-49e9-968a-2a4cff8365ae" />

<img width="770" height="488" alt="Image" src="https://github.com/user-attachments/assets/5438ec49-334c-4797-bbb8-1352f3f130f8" />


- [x] **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

Lexical Overlap vs Semantic Representation

<img width="831" height="252" alt="Image" src="https://github.com/user-attachments/assets/4c315843-2931-4b06-8439-7a0041d80afc" />

Comparative Statistical Distribution: Abstractive vs. Extractive Baseline

<img width="826" height="253" alt="Image" src="https://github.com/user-attachments/assets/533b0164-2f4c-46d7-bd02-60af5f285b0e" />
        
- [x] **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부

<img width="774" height="407" alt="Image" src="https://github.com/user-attachments/assets/bfe84514-fa50-42df-9523-cf6005f20c61" />
        
- [x] **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
     
<img width="730" height="396" alt="Image" src="https://github.com/user-attachments/assets/1851b16c-3c03-4c20-a057-a994ff7f0d82" />

# 회고(참고 링크 및 코드 개선)

각 단계별로 개념 정리 + 코드 + 결과해석 순으로 체계적인 분석 진행을 했다. 비교 분석 과정에서 단순 정성적인 평가에 그치지 않고 
정량적으로 평가까지 잘 정리를 했다. 원인(구조적 문제나 hallucination 같은..)을 해결하기 위한 개선 방안 설명도 잘 작성했다