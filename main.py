# main.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE  # ⭐ SMOTE 임포트
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
from src.xgb_trainer import XGBTrainer
from src.evaluator import ModelEvaluator


def run_analysis():
    # --- 환경 설정 ---
    data_file = 'data/raw/금융데이터셋(Kospi)__673(version 2).xls'
    target_col = 'KIS 신용평점/0A3010'

    # ----------------------------------------------------
    # 단계 1: 데이터 전처리 (3개 그룹 데이터 로드)
    # ----------------------------------------------------
    print("=============================================")
    print("단계 1: 데이터 전처리")
    print("=============================================")
    processor = DataProcessor(data_file, target_col)
    data_sets = processor.load_and_preprocess()

    if not data_sets: return

    # 3개 그룹 데이터 사용
    X_train, X_test, y_train, y_test = data_sets['AGGREGATED_SPLIT']

    # ----------------------------------------------------
    # 단계 2: SMOTE 오버샘플링 적용 (핵심)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: SMOTE 오버샘플링 적용 (Train 데이터만)")
    print("=============================================")

    print(f"SMOTE 적용 전 데이터 분포:\n{y_train.value_counts().sort_index()}")

    # SMOTE 객체 생성
    smote = SMOTE(random_state=42)

    # 훈련 데이터에만 적용 (테스트 데이터는 건드리면 안 됨!)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"\nSMOTE 적용 후 데이터 분포 (균형 맞춰짐):\n{y_train_res.value_counts().sort_index()}")

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 (SMOTE 데이터로 학습)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (SMOTE 적용)")
    print("=============================================")

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_features': [0.5, 0.7]
    }

    rf_trainer = RFTrainer()
    # ⭐ 중요: 오버샘플링된 데이터(res)를 넣습니다.
    best_rf_model, _ = rf_trainer.tune_and_train(X_train_res, y_train_res, rf_param_grid)
    rf_trainer.save_model(best_rf_model, 'models/rf_smote_3groups.pkl')

    # 평가 (평가는 원본 Test 데이터로 해야 함!)
    target_names = ['우량(0)', '보통(1)', '불량(2)']
    evaluator = ModelEvaluator(target_names=target_names)

    print("\n--- Random Forest (SMOTE) 평가 ---")
    evaluator.evaluate_model(best_rf_model, X_test, y_test, model_name="RF with SMOTE")

    # ----------------------------------------------------
    # 단계 4: XGBoost (SMOTE 데이터로 학습)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: XGBoost 튜닝 (SMOTE 적용)")
    print("=============================================")

    xgb_trainer = XGBTrainer()

    # XGBoost 파라미터 (No L2로 비교)
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'reg_lambda': [0]  # L2 없음
    }

    # ⭐ 중요: 오버샘플링된 데이터(res)를 넣습니다.
    best_xgb_model, _ = xgb_trainer.tune_and_train(X_train_res, y_train_res, xgb_param_grid)

    print("\n--- XGBoost (SMOTE) 평가 ---")
    evaluator.evaluate_model(best_xgb_model, X_test, y_test, model_name="XGB with SMOTE")

    print("\n=============================================")
    print("분석 완료. SMOTE가 수동 가중치(이전 실험)보다 더 좋은 결과를 냈는지 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()