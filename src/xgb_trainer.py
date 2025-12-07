# src/xgb_trainer.py
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight

class XGBTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def tune_and_train(self, X_train, y_train, param_grid, scoring='recall_weighted'):
        print("\n--- XGBoost L2 정규화 하이퍼파라미터 튜닝 (Grid Search) 시작 ---")

        # ⚠️ 중요: XGBoost는 레이블이 0부터 시작해야 합니다. (1~10등급 -> 0~9등급)
        # 데이터가 이미 0부터 시작하는지 확인 후 처리
        if y_train.min() > 0:
            y_train_shifted = y_train - 1
        else:
            y_train_shifted = y_train

        # 10개 등급 불균형 해소를 위한 샘플 가중치 계산
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_shifted)

        # 기본 모델 정의 (n_jobs=1로 설정하여 충돌 방지)
        xgb_model = XGBClassifier(
            random_state=self.random_state,
            objective='multi:softmax',
            num_class=10,  # 10개 등급
            eval_metric='mlogloss',
            n_jobs=1
        )

        # Grid Search 정의
        xgb_grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=3,
            verbose=1,
            n_jobs=1
        )

        # 학습 (sample_weight 전달)
        xgb_grid_search.fit(X_train, y_train_shifted, sample_weight=sample_weights)

        best_xgb_model = xgb_grid_search.best_estimator_
        best_xgb_params = xgb_grid_search.best_params_
        best_xgb_score = xgb_grid_search.best_score_

        print("\n--- XGBoost 최적 파라미터 및 성능 ---")
        print(f"최적의 파라미터: {best_xgb_params}")
        print(f"최적 모델의 가중평균 Recall: {best_xgb_score:.4f}")

        return best_xgb_model, best_xgb_params

    def save_model(self, model, path):
        joblib.dump(model, path)
        print(f"XGBoost 모델 저장 완료: {path}")