# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최에리나
- 리뷰어 : 최에리나


# PRT(Peer Review Template)
- [x]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
        <img width="866" height="552" alt="Screenshot 2026-05-19 at 11 58 32 AM" src="https://github.com/user-attachments/assets/6eb2bf22-10ba-4c75-a8a4-78199bb376ad" />

- [x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        <img width="1019" height="472" alt="Screenshot 2026-05-19 at 11 20 50 AM" src="https://github.com/user-attachments/assets/acd8a1df-2222-44c6-b902-aea551ae3af4" />

- [x]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        <img width="1874" height="935" alt="Screenshot 2026-05-19 at 11 21 37 AM" src="https://github.com/user-attachments/assets/1de2c6b8-87ef-4a71-9804-0f6fbd3baa03" />

- [x]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        <img width="1015" height="785" alt="Screenshot 2026-05-19 at 11 22 03 AM" src="https://github.com/user-attachments/assets/af8fdc01-f78c-48dc-9672-60fa7f4d4b4c" />

- [x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        <img width="1100" height="688" alt="Screenshot 2026-05-19 at 11 57 39 AM" src="https://github.com/user-attachments/assets/d2e882fd-27aa-42b6-ae21-c2ca0e005dba" />


# 회고(참고 링크 및 코드 개선)
BLEU score가 기대보다 낮게 나왔고, 생성된 문장도 전반적인 문맥은 맞았지만 번역의 완성도는 다소 아쉬웠다. 단순히 epoch 수를 늘리기보다는 regularization 강도를 조정하는 등 보다 세심한 하이퍼파라미터 튜닝이 필요해 보인다. 또한, Greedy Decoding 외에도 Beam Search를 적용하거나, Mecab과 SentencePiece를 결합한 하이브리드 토큰화를 시도해보는 것도 성능 개선에 도움이 될 것 같다.

