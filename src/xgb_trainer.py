import joblib
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight  # 이 모듈은 L2 튜닝에서는 사용되지 않지만, 오류 수정에는 무관합니다.


class XGBTrainer:
    """
    XGBoost 모델의 L2 정규화 (reg_lambda) 튜닝, 학습 및 저장을 담당하는 클래스.
    """

    def tune_and_train(self, X_train, y_train, param_grid, scoring='recall_weighted'):
        print("\n--- XGBoost L2 정규화 하이퍼파라미터 튜닝 (Grid Search) 시작 ---")

        # 기본 모델 정의
        # [수정] n_jobs=1로 설정하여 멀티프로세싱 충돌 방지
        xgb_model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=1  # 치명적인 오류 방지를 위해 병렬 처리 비활성화
        )

        # Grid Search 정의: L2 정규화 파라미터(reg_lambda)를 포함하여 튜닝
        # [수정] n_jobs=1로 설정하여 Grid Search의 병렬 처리 충돌 방지
        xgb_grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring=scoring,
            cv=3,
            verbose=1,
            n_jobs=1  # 충돌 방지를 위해 n_jobs=-1 대신 1 사용
        )

        # ... (이하 코드는 동일)
        xgb_grid_search.fit(X_train, y_train)
        # ...

        best_xgb_model = xgb_grid_search.best_estimator_
        best_xgb_params = xgb_grid_search.best_params_
        best_xgb_score = xgb_grid_search.best_score_

        print("\n--- 최적 파라미터 및 성능 ---")
        print(f"최적의 파라미터: {best_xgb_params}")
        print(f"최적 모델의 가중평균 Recall: {best_xgb_score:.4f}")

        return best_xgb_model, best_xgb_params

    def save_model(self, model, path):
        """최적화된 모델을 .pkl 파일로 저장"""
        joblib.dump(model, path)
        print(f"모델 저장 완료: {path}")