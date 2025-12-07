# main.py
import pandas as pd
import numpy as np
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
    # 단계 1: 데이터 전처리 및 1차 변수 선정 (VIF)
    # ----------------------------------------------------
    print("=============================================")
    print("단계 1: 데이터 전처리 및 1차 변수 선정 (VIF)")
    print("=============================================")
    processor = DataProcessor(data_file, target_col)
    data_sets = processor.load_and_preprocess()

    if not data_sets:
        print("분석을 계속할 수 없습니다.")
        return

    # ⭐ 비교 실험을 위해 '원본 10개 등급' 사용 ⭐
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = data_sets['ORIGINAL_SPLIT']

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (원본 10등급 사용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (원본 10등급 사용)")
    print("=============================================")

    logreg_trainer = LogRegTrainer()
    _, insignificant_vars = logreg_trainer.train_ordinal_model(X_train_orig, y_train_orig)

    if insignificant_vars:
        print(f"\n--> 2차 변수 선정 적용: {len(insignificant_vars)}개 변수 제거 ({insignificant_vars})")
        X_train_final = X_train_orig.drop(columns=insignificant_vars, errors='ignore')
        X_test_final = X_test_orig.drop(columns=insignificant_vars, errors='ignore')
    else:
        print("\n--> 2차 변수 선정에서 추가 제거 변수 없음.")
        X_train_final = X_train_orig
        X_test_final = X_test_orig

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 튜닝 (비교 기준점)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (Baseline)")
    print("=============================================")

    rf_param_grid = {'n_estimators': [100], 'max_features': [0.5]}  # 빠른 실행을 위해 고정
    rf_trainer = RFTrainer()
    best_rf_model, _ = rf_trainer.tune_and_train(X_train_final, y_train_orig, rf_param_grid)

    target_names_10 = [str(i) for i in sorted(y_train_orig.unique())]
    evaluator = ModelEvaluator(target_names=target_names_10)
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test_orig, model_name="Random Forest")

    # ----------------------------------------------------
    # ⭐ 단계 5-A: XGBoost (L2 미적용) ⭐
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 5-A: XGBoost (L2 규제 없음, reg_lambda=0)")
    print("=============================================")

    xgb_trainer = XGBTrainer()

    # reg_lambda = 0 으로 설정하여 규제 제거
    xgb_params_no_l2 = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'reg_lambda': [0]  # <--- L2 끄기
    }

    best_xgb_no_l2, _ = xgb_trainer.tune_and_train(X_train_final, y_train_orig, xgb_params_no_l2)
    xgb_trainer.save_model(best_xgb_no_l2, 'models/xgboost_no_l2.pkl')

    # 평가 (0~9 변환)
    y_test_shifted = y_test_orig - 1 if y_test_orig.min() > 0 else y_test_orig
    evaluator.evaluate_model(best_xgb_no_l2, X_test_final, y_test_shifted, model_name="XGBoost (No L2)")

    # ----------------------------------------------------
    # ⭐ 단계 5-B: XGBoost (L2 적용) ⭐
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 5-B: XGBoost (L2 규제 적용, reg_lambda 튜닝)")
    print("=============================================")

    # reg_lambda를 여러 값으로 실험
    xgb_params_l2 = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'reg_lambda': [1.0, 10.0, 50.0]  # <--- L2 켜기 (강도 조절)
    }

    best_xgb_l2, best_params_l2 = xgb_trainer.tune_and_train(X_train_final, y_train_orig, xgb_params_l2)
    xgb_trainer.save_model(best_xgb_l2, 'models/xgboost_with_l2.pkl')

    print(f"\n>> 결정된 최적의 L2 규제 강도: {best_params_l2['reg_lambda']}")

    evaluator.evaluate_model(best_xgb_l2, X_test_final, y_test_shifted, model_name="XGBoost (With L2)")

    print("\n=============================================")
    print("분석 완료. [No L2] vs [With L2]의 불량 등급 Recall을 비교하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()