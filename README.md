# Crossfit-3D 프로젝트
(2D 데이터셋에서의 성능 개선을 위해 z축을 추가한 3D 데이터셋으로 진행한 **최종 제안**프로젝트입니다.)

# 전체 크로스핏 데이터셋 분석 프로세스 순서도(Repository명)
initial-approach(초기 접근법) -> new-approach-2d(2D 데이터셋 개선) -> new-approach-3d(차원 확장)

## 1. 프로젝트 개요
크로스핏 운동 중 4가지 핵심 동작인 Deadlift, Press, Squat, Clean에서 발생하는 정상 자세와 오류자세를 분류하는 딥러닝 기반 시스템을 개발하는 것을 목표로 합니다.


## 2. 실험 코드 제출

### 주요 파일 구조

```
├── README.md                          # 프로젝트 설명서
├── crossfit_plus_3d.ipynb            # 메인 실험 노트북
├── clean.csv                          # Clean 운동 데이터
├── deadlift.csv                       # Deadlift 운동 데이터  
├── press.csv                          # Press 운동 데이터
├── squat.csv                          # Squat 운동 데이터
├── best_clean_cnn_model.h5           # Clean 운동 최적 CNN 모델
├── best_deadlift_cnn_model.h5        # Deadlift 운동 최적 CNN 모델
├── best_press_cnn_lstm_model.h5      # Press 운동 최적 CNN-LSTM 모델
├── best_squat_cnn_model.h5           # Squat 운동 최적 CNN 모델
├── clean_metadata.pkl                # Clean 모델 메타데이터 (피처명, 클래스 정보 등)
├── clean_scaler.pkl                  # Clean 모델용 StandardScaler/MinMaxScaler 객체
├── deadlift_metadata.pkl             # Deadlift 모델 메타데이터 (피처명, 클래스 정보 등)
├── deadlift_scaler.pkl               # Deadlift 모델용 StandardScaler/MinMaxScaler 객체
├── press_metadata.pkl                # Press 모델 메타데이터 (피처명, 클래스 정보 등)
├── press_scaler.pkl                  # Press 모델용 StandardScaler/MinMaxScaler 객체
├── squat_metadata.pkl                # Squat 모델 메타데이터 (피처명, 클래스 정보 등)
├── squat_scaler.pkl                  # Squat 모델용 StandardScaler/MinMaxScaler 객체
└── experiment_results.pkl            # 실험 결과 데이터 (성능 지표, 하이퍼파라미터 등)
```

### 코드 실행 방법

1. **환경 설정**
   - 구글 코랩 A100 GPU사용 (코랩에서 csv파일만 런타임에 넣어주신다면 그대로 실행하셔도 됩니다!)
   - Python 3.11.13
   - 필요한 라이브러리: tensorflow, pandas, numpy, scikit-learn, matplotlib등 (버전은 requirements.txt에 명시해두었습니다.)

3. **실행 순서**
   ```bash
   # Jupyter Notebook 실행
   jupyter notebook crossfit_plus_3d.ipynb
   ```

4. **노트북 실행 가이드**
   - 셀 단위로 순차적으로 실행
   - 각 운동별 데이터 로드 및 전처리
   - 모델 학습 및 평가
   - 결과 시각화 및 분석

### 데이터셋 정보

#### 데이터 출처
본 프로젝트에서 사용된 CrossFit 운동 데이터셋은 다음 소스를 기반으로 합니다:

**원본 데이터셋:**
- **데이터셋 명**: 크로스핏 동작 데이터셋
- **출처 URL**: [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%ED%81%AC%EB%A1%9C%EC%8A%A4%ED%95%8F%20%EB%8F%99%EC%9E%91%20%EB%8D%B0%EC%9D%B4%ED%84%B0&aihubDataSe=data&dataSetSn=71422]
  
#### 데이터 전처리

### 모델 정보

#### Clean 운동 모델 (best_clean_cnn_model.h5)
#### Deadlift 운동 모델 (best_deadlift_cnn_model.h5)
#### Press 운동 모델 (best_press_cnn_lstm_model.h5)
#### Squat 운동 모델 (best_squat_cnn_model.h5)

### 실험 결과

실험 결과는 `experiment_results.pkl` 파일에 저장되어 있으며, 다음 내용을 포함합니다:
- 각 모델의 성능 지표 (정확도, 손실, MSE 등)
- 교차 검증 결과
- 하이퍼파라미터 튜닝 기록
- 학습 곡선 데이터

### 모델 메타데이터 (PKL 파일들)

#### Scaler 파일들 (*.scaler.pkl)
각 운동별 데이터 전처리에 사용된 스케일러 객체:
- **clean_scaler.pkl**: Clean 운동 데이터 정규화용 스케일러
- **deadlift_scaler.pkl**: Deadlift 운동 데이터 정규화용 스케일러  
- **press_scaler.pkl**: Press 운동 데이터 정규화용 스케일러
- **squat_scaler.pkl**: Squat 운동 데이터 정규화용 스케일러

```python
# 사용 예시
import pickle
with open('clean_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    normalized_data = scaler.transform(raw_data)
```

#### Metadata 파일들 (*.metadata.pkl)
각 모델의 메타데이터 정보:
- **피처 이름 목록**: 입력 데이터의 각 컬럼명
- **클래스 정보**: 분류 모델의 경우 클래스 라벨
- **데이터 통계**: 평균, 표준편차, 최솟값, 최댓값 등
- **전처리 파라미터**: 정규화 범위, 필터링 조건 등
- **모델 하이퍼파라미터**: 학습 시 사용된 설정값

```python
# 사용 예시
import pickle
with open('clean_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)
    feature_names = metadata['feature_names']
    class_labels = metadata['class_labels']
    preprocessing_params = metadata['preprocessing']
```

1. **데이터 전처리**: 각 운동별 스케일러(scaler.pkl)와 메타데이터(metadata.pkl) 파일이 필요
2. **실행 환경**: CUDA 지원 GPU 환경에서 실행 시 학습 속도 향상
3. **메모리 요구사항**: 대용량 데이터셋 처리를 위해 충분한 RAM 필요
