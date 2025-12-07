# src/rf_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


class RFTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def tune_and_train(self, X_train, y_train, param_grid):
        print("\n--- 랜덤 포레스트 하이퍼파라미터 튜닝 (Grid Search) 시작 ---")

        # ⭐⭐ 수정: 10개 등급 처리를 위해 'balanced' 자동 가중치로 원복 ⭐⭐
        rf_base = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')

        # Grid Search 객체 생성
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            scoring='recall_weighted',
            cv=3,
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print("\n--- 최적 파라미터 및 성능 ---")
        print(f"최적의 파라미터: {grid_search.best_params_}")
        print(f"최적 모델의 가중평균 Recall: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model, model_path):
        joblib.dump(model, model_path)
        print(f"모델 저장 완료: {model_path}")