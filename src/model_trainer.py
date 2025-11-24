# src/model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


class ModelTrainer:
    def __init__(self, n_estimators=150, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def train_model(self, X_train, y_train):
        print("[2-1] 모델 훈련 시작 (Random Forest, class_weight='balanced')")

        # ⭐ 민감도 조정 (불균형 해소) 핵심 파라미터 적용 ⭐
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight='balanced'  # <--- 소수 클래스에 자동 가중치 부여 (Recall 향상 목표)
        )

        self.model.fit(X_train, y_train)
        print("모델 훈련 완료.")

        return self.model

    def save_model(self, model, model_path):
        """훈련된 모델을 파일로 저장"""
        joblib.dump(model, model_path)
        print(f"[2-2] 모델 저장 완료: {model_path}")