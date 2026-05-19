# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 최에리나
- 리뷰어 : 송세미


# PRT(Peer Review Template)
- [X]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 한국어-영어 병렬 말뭉치(korean-english-park)의 데이터 클리닝과 전처리 단계가 유기적으로 잘 구현되어 있다. SentencePiece 토크나이저를 활용하여 Vocabulary Size 10,000 규모의 단어 사전을 성공적으로 빌드했고 데이터셋 다운로드 및 분할(Train 94,123 / Val 1,000 / Test 2,000)이 누락 없이 완료되었다.

모델 학습 루프 및 결과 리포팅 구조가 최종 단계까지 에러 없이 완벽하게 구축되어 첨부되었다.
    
- [X]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
   Excellent. 코드 전반에 걸쳐 하이퍼파라미터의 차원 설정 이유가 상세하게 기술되어있다. 독립적인 임베딩 차원(EMB_DIM = 256)과 GRU의 은닉 차원(HID_DIM = 512)이 왜 다른지, 문맥 처리 역량 관점에서 정량적·개념적 주석이 잘 달아져있다.
<img width="940" height="531" alt="image" src="https://github.com/user-attachments/assets/c242e790-580b-46c5-a567-4155c0e0a992" />

Bahdanau Attention 연산에서 발생할 수 있는 Query(Q), Key(K), Value(V)의 관계를 뉴스 도메인 예시("이탈리아 천문학자들이 행성의 대기에서 물을 발견했다.")를 들어 수식과 매핑 구조로 풀어낸 마크다운 및 주석 설명은 코드의 가독성을 극대화해준다. 
<img width="920" height="654" alt="image" src="https://github.com/user-attachments/assets/a55a6ad6-1d0c-4932-8b86-f09dbfc98a45" />

클래식 Seq2Seq RNN 모델에서는 K=V=\text{Encoder Hidden States}가 성립하는 구조적 특징을 짚어낸 부분이 좋다.
        
- [X]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    Good. 일반적인 CUDA 가속 환경 외에도 환경에서 연산 속도를 극대화할 수 있도록 torch.backends.mps.is_available() 분기문을 적용하여 MPS 장치 가속 설정을 매끄럽게 처리한 기록이 보여진다.

clean_text 함수를 정의할 때 한국어 정규식([^ㄱ-ㅎ가-힣a-zA-Z?.!, ]+)과 영어 소문자 변환 로직(is_korean=False)을 다르게 타겟팅하여, 데이터 클리닝 단계에서 발생할 수 있는 토크나이징 예외 및 노이즈 인입 문제를 사전에 효율적으로 디버깅 및 차단한걸로 보여진다.
<img width="830" height="283" alt="image" src="https://github.com/user-attachments/assets/008d75e6-cd29-4e1b-8984-8a417f945919" />

        
- [X]  **4. 회고를 잘 작성했나요?**
    Excellent. 단순한 수치 나열에 그치지 않고 NMT 모델의 발전사(Seq2Seq \rightarrow LSTM/GRU \rightarrow Bahdanau Attention \rightarrow Transformer Self-Attention)를 이론적 흐름으로 일목요연하게 표로 정리해 둔 회고 서술 방식이 좋았다.
<img width="910" height="443" alt="image" src="https://github.com/user-attachments/assets/47d0aa6e-d8b6-4b5a-aef2-370eb96867c0" />
모델 인퍼런스 시 노출되는 Exposure Bias 문제와, 해결하기 위한 Teacher Forcing 기법의 한계점을 향후 Strategic Actions와 유기적으로 연계해 둔 분석적 고찰이 매우 깊이있다.
<img width="997" height="232" alt="image" src="https://github.com/user-attachments/assets/7e7df857-9f19-4b08-b469-637f05232c96" />

        
- [X]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드(PEP8)의 정석을 따르고 있으며, 전처리 파이프라인(clean_text, load_and_preprocess)이 고도로 모듈화되어 중복 코드를 최소화했다. 불필요하게 복잡한 서드파티 라이브러리의 인입을 제한하고 sentencepiece와 내장 패키지만을 조합하여 깔끔하고 가벼운 전처리 아키텍처를 구현했다.
    - <img width="923" height="532" alt="image" src="https://github.com/user-attachments/assets/e8389e47-f3cb-4ae4-b89c-754fadeba9f7" />


# 회고(참고 링크 및 코드 개선)
```
이론적 배경 정리부터 하드웨어 가속기(MPS) 설정, 꼼꼼한 다국어 데이터 클리닝까지 흠잡을 곳 없이 잘 작성된 프로젝트였다. 딥러닝 아키텍처에 대한 이해도가 코드 한 줄 한 줄에 그대로 투영되어 있어 리뷰하는 동안 많은 것을 배울수 있었다. 결론부에 정리해 주신 Exposure Bias 분석이 제일 인상 깊었다.

마지막 부분에 작성한 "Strategic actions" 중에서 Advanced Decoding Strategies와 관련된 아이디어를 실제 추론 루프로 확장할 수 있는 참고할 만한 아이디어 한가지를 추가해본다.  현재 구현된 Greedy Decoding 방식은 단어가 반복 출력되는 오차에 취약할 수 있으므로, 아래와 같은 구조로 Beam Search 추론 루프를 도입해 보면 번역 퀄리티가 한층 더 상승할수 있겠다.

# Greedy Decoding을 보완할 Beam Search 개념 구조 제안
def beam_search_decode(model, src_tensor, beam_size=3, max_len=50):
    encoder_outputs, hidden = model.encoder(src_tensor)

    # 첫 번째 빔 시작 세팅 (확률 로그값, 토큰 리스트, 디코더 hidden)
    start_token = [<s>_id]
    beams = [(0.0, start_token, hidden)]

    # 최대 길이까지 상위 top-k개의 빔 경로를 유지하며 확장 수행
    pass

```
