# src/xgb_trainer.py
from xgboost import XGBClassifier
import joblib
from sklearn.utils.class_weight import compute_sample_weight


class XGBTrainer:
    def __init__(self, n_estimators=150, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None

    def train_model(self, X_train, y_train):
        print("[2-1] 모델 훈련 시작 (XGBoost, class_weight='balanced')")

        # XGBoost는 class_weight='balanced'를 직접 지원하지 않으므로,
        # Scikit-learn의 compute_sample_weight를 이용해 가중치를 계산합니다.
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            objective='multi:softmax',
            n_jobs=-1  # 모든 코어 사용
        )

        # 계산된 가중치를 fit 메서드에 전달합니다. (Cost-Sensitive Learning 구현)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        print("모델 훈련 완료.")

        return self.model

    def save_model(self, model, model_path):
        """훈련된 모델을 파일로 저장"""
        joblib.dump(model, model_path)
        print(f"[2-2] 모델 저장 완료: {model_path}")