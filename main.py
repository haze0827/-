# main.py
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
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

    # ⭐⭐ 핵심 변경: 원본 10개 등급 데이터셋만 사용합니다. ⭐⭐
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

    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_features': [0.5, 0.7]
    }

    rf_trainer = RFTrainer()
    # ⭐⭐ 핵심 변경: y_train_orig (10개 등급)를 전달합니다. ⭐⭐
    best_rf_model, best_params = rf_trainer.tune_and_train(X_train_final, y_train_orig, param_grid)

    model_save_path = 'models/random_forest_final_10grades.pkl'
    rf_trainer.save_model(best_rf_model, model_save_path)

    # ----------------------------------------------------
    # 단계 4: 최종 선정 모델 평가
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 (10개 등급)")
    print("=============================================")

    # ⭐⭐ 핵심 변경: 10개 등급 레이블 이름 생성 ⭐⭐
    target_names = [str(i) for i in sorted(y_train_orig.unique())]
    evaluator = ModelEvaluator(target_names=target_names)

    # ⭐⭐ 핵심 변경: y_test_orig (10개 등급)를 사용하여 평가합니다. ⭐⭐
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test_orig, model_name="최종 Random Forest (10등급)")

    print("\n=============================================")
    print("분석 완료. 8, 9, 10등급(불량)의 Recall이 0.00인지 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()