# main.py
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
from src.xgb_trainer import XGBTrainer  # <-- [추가]
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

    # 원본 10개 등급 데이터셋 사용
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = data_sets['ORIGINAL_SPLIT']

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (원본 10등급 사용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (원본 10등급 사용)")
    print("=============================================")

    logreg_trainer = LogRegTrainer()
    _, insignificant_vars = logreg_trainer.train_ordinal_model(X_train_orig, y_train_orig)

    # 2차 변수 선정 결과 적용
    if insignificant_vars:
        print(f"\n--> 2차 변수 선정 적용: {len(insignificant_vars)}개 변수 제거 ({insignificant_vars})")
        X_train_final = X_train_orig.drop(columns=insignificant_vars, errors='ignore')
        X_test_final = X_test_orig.drop(columns=insignificant_vars, errors='ignore')
    else:
        print("\n--> 2차 변수 선정에서 추가 제거 변수 없음.")
        X_train_final = X_train_orig
        X_test_final = X_test_orig

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 튜닝 (원본 10개 등급 사용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (원본 10개 등급 사용)")
    print("=============================================")

    rf_param_grid = {
        'n_estimators': [100, 150],
        'max_features': [0.5, 0.7]
    }

    rf_trainer = RFTrainer()
    best_rf_model, _ = rf_trainer.tune_and_train(X_train_final, y_train_orig, rf_param_grid)
    rf_trainer.save_model(best_rf_model, 'models/random_forest_final_10grades.pkl')

    # ----------------------------------------------------
    # 단계 4: 랜덤 포레스트 평가
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: 랜덤 포레스트 모델 평가")
    print("=============================================")

    target_names_10 = [str(i) for i in sorted(y_train_orig.unique())]
    evaluator = ModelEvaluator(target_names=target_names_10)
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test_orig, model_name="Random Forest (10등급)")

    # ----------------------------------------------------
    # 단계 5: XGBoost 튜닝 및 평가 (원본 10개 등급 사용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 5: XGBoost 튜닝 및 평가 (원본 10개 등급 사용)")
    print("=============================================")

    xgb_trainer = XGBTrainer()

    # XGBoost 튜닝 범위 (간소화)
    xgb_param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.1]
    }

    # XGBoost 훈련 (튜닝 포함)
    best_xgb_model, _ = xgb_trainer.tune_and_train(X_train_final, y_train_orig, xgb_param_grid)
    xgb_trainer.save_model(best_xgb_model, 'models/xgboost_final_10grades.pkl')

    # XGBoost 평가 (주의: y_test도 0~9로 변환해서 평가해야 함)
    print("\n--- XGBoost (10등급) 최종 평가 ---")

    # y_test 데이터도 0~9로 변환 (10등급인 경우 1을 뺌)
    if y_test_orig.min() > 0:
        y_test_shifted = y_test_orig - 1
    else:
        y_test_shifted = y_test_orig

    # 평가 시 target_names는 그대로 사용 (의미는 1~10등급이므로)
    evaluator.evaluate_model(best_xgb_model, X_test_final, y_test_shifted, model_name="XGBoost (10등급)")

    print("\n=============================================")
    print("분석 완료. RF와 XGBoost 모두 9, 10등급 Recall이 낮은지 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()