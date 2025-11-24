# src/evaluator.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


class ModelEvaluator:
    def __init__(self, target_names=None):
        self.target_names = target_names

    def evaluate_model(self, model, X_test, y_test, model_name="모델"):
        print(f"\n--- {model_name} 성능 평가 시작 ---")

        y_pred = model.predict(X_test)

        # 실제 존재하는 레이블만 사용 (Zero-Indexing 적용된 0, 1, 2)
        labels = sorted(y_test.unique())

        # 1. Confusion Matrix 출력
        print("\n--- 혼동 행렬 (Confusion Matrix) ---")
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        print(cm)

        # 2. Classification Report 출력 (Recall/Precision 확인)
        print("\n--- 등급별 상세 리포트 (Classification Report) ---")
        # target_names는 main.py에서 정의된 '고위험(0)', '중위험(1)', '저위험(2)'가 사용됩니다.
        report = classification_report(y_test, y_pred, labels=labels, target_names=self.target_names, zero_division=0)
        print(report)

        # 3. 전체 정확도 출력 (참고용)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n전체 정확도 (Accuracy): {accuracy:.4f}")

        return report, cm