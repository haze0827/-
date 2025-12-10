# main.py
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
from src.xgb_trainer import XGBTrainer  # XGBTrainer 임포트 필수
from src.evaluator import ModelEvaluator


def run_analysis():
    # --- 환경 설정 ---
    data_file = 'data/raw/금융데이터셋(Kospi)__673(version 2).xls'
    target_col = 'KIS 신용평점/0A3010'

    # ----------------------------------------------------
    # 단계 1: 데이터 전처리
    # ----------------------------------------------------
    print("=============================================")
    print("단계 1: 데이터 전처리 (파생변수 -> 3개 그룹)")
    print("=============================================")
    processor = DataProcessor(data_file, target_col)
    data_sets = processor.load_and_preprocess()

    if not data_sets: return

    # ⭐ 3개 그룹 통합 데이터셋 사용 ⭐
    X_train, X_test, y_train, y_test = data_sets['AGGREGATED_SPLIT']

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (변수 선정용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (변수 검증)")
    print("=============================================")

    logreg_trainer = LogRegTrainer()
    _, insignificant_vars = logreg_trainer.train_ordinal_model(X_train, y_train)

    if insignificant_vars:
        print(f"\n--> 2차 변수 선정 적용: {len(insignificant_vars)}개 변수 제거")
        X_train_final = X_train.drop(columns=insignificant_vars, errors='ignore')
        X_test_final = X_test.drop(columns=insignificant_vars, errors='ignore')
    else:
        X_train_final = X_train
        X_test_final = X_test

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 (3개 그룹, 수동 가중치)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (3개 그룹)")
    print("=============================================")

    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_features': [0.5, 0.7]
    }

    rf_trainer = RFTrainer()
    best_rf_model, _ = rf_trainer.tune_and_train(X_train_final, y_train, rf_param_grid)

    # 평가용 레이블 이름
    target_names_3groups = ['우량(0)', '보통(1)', '불량(2)']
    evaluator = ModelEvaluator(target_names=target_names_3groups)

    print("\n--- Random Forest (3그룹) 평가 ---")
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test, model_name="Random Forest")

    # ----------------------------------------------------
    # ⭐ 단계 4: XGBoost (3개 그룹, L2 규제 없음) ⭐
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: XGBoost (3개 그룹, L2 규제 없음)")
    print("=============================================")

    xgb_trainer = XGBTrainer()

    # reg_lambda = 0 (L2 규제 끔)
    xgb_params_no_l2 = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'reg_lambda': [0]  # <--- 핵심: L2 미적용
    }

    best_xgb_model, _ = xgb_trainer.tune_and_train(X_train_final, y_train, xgb_params_no_l2)

    print("\n--- XGBoost (3그룹, No L2) 평가 ---")
    # XGBoost는 0, 1, 2 레이블을 그대로 사용하므로 y_test 변환 불필요 (이미 0, 1, 2임)
    evaluator.evaluate_model(best_xgb_model, X_test_final, y_test, model_name="XGBoost (No L2)")

    print("\n=============================================")
    print("분석 완료. RF(수동 가중치)와 XGB(No L2) 중 불량(2) Recall이 높은 것은?")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()