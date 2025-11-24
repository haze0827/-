# src/rf_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib


class RFTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def tune_and_train(self, X_train, y_train, param_grid):
        print("\n--- 랜덤 포레스트 하이퍼파라미터 튜닝 (Grid Search) 시작 ---")

        # class_weight='balanced'를 적용하여 불균형 문제를 해결합니다.
        rf_base = RandomForestClassifier(random_state=self.random_state, class_weight='balanced')

        # Grid Search 객체 생성
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            scoring='recall_weighted',  # 불균형 데이터에 맞게 Recall을 최적화 목표로 설정
            cv=3,  # 3-fold 교차 검증
            verbose=1,
            n_jobs=-1  # 모든 코어 사용
        )

        # y_train은 3개 그룹(0, 1, 2)으로 통합된 데이터입니다.
        grid_search.fit(X_train, y_train)

        print("\n--- 최적 파라미터 및 성능 ---")
        print(f"최적의 파라미터: {grid_search.best_params_}")
        print(f"최적 모델의 가중평균 Recall: {grid_search.best_score_:.4f}")

        # 최종 모델 반환
        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model, model_path):
        """훈련된 모델을 파일로 저장"""
        joblib.dump(model, model_path)
        print(f"모델 저장 완료: {model_path}")