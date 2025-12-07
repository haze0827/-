# src/logreg_trainer.py
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np


class LogRegTrainer:
    def __init__(self):
        pass

    def train_ordinal_model(self, X_train, y_train):
        print("\n--- 순서형 로지스틱 회귀분석 (Ordered Model) 시작 ---")

        # NumPy 배열로 강제 변환 (ValueError 방지)
        X_array = X_train.values
        y_array = y_train.values

        # 원본 변수명 리스트 저장 (나중에 복원용)
        feature_names = list(X_train.columns)

        # 모델 훈련
        ord_mod = OrderedModel(y_array, X_array, distr='logit')
        res_log = ord_mod.fit(method='bfgs', maxiter=5000)

        # ⭐⭐⭐ 변수명 복원 및 요약 출력 ⭐⭐⭐
        try:
            # 변수명 리스트 생성 (특징 변수명 + Threshold 이름)
            num_params = len(res_log.params)
            num_features = len(feature_names)
            num_thresholds = num_params - num_features

            # Threshold 이름 생성 (예: 0/1, 1/2...)
            threshold_names = [f"{i}/{i + 1}" for i in range(num_thresholds)]

            # 전체 이름 리스트 (특징 + Threshold)
            full_names = feature_names + threshold_names

            # 변수명이 포함된 요약 출력
            print(res_log.summary(xname=full_names))

        except Exception as e:
            print(f"변수명 매핑 중 경고 발생 (기본 출력 사용): {e}")
            print(res_log.summary())

        # ⭐⭐⭐ 2차 변수 선정: p-value(유의확률) > 0.05 변수 확인 ⭐⭐⭐

        # 1. pvalues 전체 배열 가져오기
        pvalues_full = res_log.pvalues

        # 2. 특징(Feature)에 해당하는 p-value만 추출 (Threshold 제외)
        feature_pvalues = pvalues_full[:len(feature_names)]

        # 3. p-value가 0.05보다 큰 변수의 인덱스 찾기
        insignificant_indices = np.where(feature_pvalues > 0.05)[0]

        # 4. 인덱스를 원본 컬럼 이름으로 매핑
        vars_to_remove = [feature_names[i] for i in insignificant_indices]

        print(f"\n2차 변수 선정 제거 후보 (p > 0.05): {vars_to_remove}")

        return res_log, vars_to_remove