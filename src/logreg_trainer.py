# src/logreg_trainer.py
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np


class LogRegTrainer:
    def __init__(self):
        pass

    def train_ordinal_model(self, X_train, y_train):
        print("\n--- 순서형 로지스틱 회귀분석 (Ordered Model) 시작 ---")

        # ⭐⭐⭐ 상수항 추가 코드를 제거했습니다. OrderedModel이 자체적으로 처리합니다. ⭐⭐⭐
        # 만약 OrderedModel이 상수항을 필요로 하지 않는다면, X_train에는 상수항이 없어야 합니다.
        ord_mod = OrderedModel(y_train, X_train, distr='logit')

        # 모델 훈련
        # Warning: 최적화에 시간이 오래 걸릴 수 있으므로 maxiter를 적당히 설정했습니다.
        res_log = ord_mod.fit(method='bfgs', maxiter=1000)

        print(res_log.summary())

        # 2차 변수 선정: p-value(유의확률) > 0.05 변수 확인
        # p-value는 모델의 신뢰도를 판단하고 2차 변수 선정에 사용됩니다.
        insignificant_vars = res_log.pvalues[res_log.pvalues > 0.05].index.tolist()

        # OrderedModel의 결과에는 'const' 대신 등급 경계(Threshold)에 해당하는 값이 포함될 수 있습니다.
        # 이를 구분하기 위해 p-value가 0.05보다 큰 변수 중 'const' 또는 'cut' 관련 변수명을 제외하는 로직이 필요합니다.

        # 현재는 X_train의 컬럼에 해당하는 변수만 제거 후보로 간주합니다.

        # pvalues에서 X_train의 컬럼명이 아닌 항목들을 제거합니다. (Thresholds)
        x_columns = list(X_train.columns)

        # 순수한 특징(Feature)에 대한 p-value만 남깁니다.
        vars_to_remove = []
        for var in insignificant_vars:
            if var in x_columns:
                vars_to_remove.append(var)

        print(f"\n2차 변수 선정 제거 후보 (p > 0.05): {vars_to_remove}")

        return res_log, vars_to_remove