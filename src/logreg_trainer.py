# src/logreg_trainer.py
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np


class LogRegTrainer:
    def __init__(self):
        pass

    def train_ordinal_model(self, X_train, y_train):
        print("\n--- 순서형 로지스틱 회귀분석 (Ordered Model) 시작 ---")

        # OrderedModel은 상수항이 필요 없습니다.
        ord_mod = OrderedModel(y_train, X_train, distr='logit')

        # 모델 훈련
        res_log = ord_mod.fit(method='bfgs', maxiter=1000)

        print(res_log.summary())

        # 2차 변수 선정: p-value(유의확률) > 0.05 변수 확인
        insignificant_vars = res_log.pvalues[res_log.pvalues > 0.05].index.tolist()

        x_columns = list(X_train.columns)

        # 순수한 특징(Feature)에 대한 p-value가 높은 변수만 제거 후보로 간주
        vars_to_remove = []
        for var in insignificant_vars:
            if var in x_columns:
                vars_to_remove.append(var)

        print(f"\n2차 변수 선정 제거 후보 (p > 0.05): {vars_to_remove}")

        return res_log, vars_to_remove