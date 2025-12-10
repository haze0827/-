# src/rf_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


class RFTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def tune_and_train(self, X_train, y_train, param_grid):
        print("\n--- 랜덤 포레스트 하이퍼파라미터 튜닝 (Grid Search) 시작 ---")

        # ⭐⭐ 핵심 전략: 3개 그룹용 수동 비용 가중치 (Cost-Sensitive Learning) ⭐⭐
        # 0(우량): 1.0 (기본)
        # 1(보통): 5.0 (중요도 높음)
        # 2(불량): 20.0 (매우 치명적 - Recall 극대화 목표)
        custom_weights = {0: 1.0, 1: 5.0, 2: 20.0}

        # 수동 가중치 적용
        rf_base = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=custom_weights
        )

        # Grid Search 설정 (Recall을 최우선으로 튜닝)
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