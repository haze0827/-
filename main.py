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
    # 단계 1: 데이터 전처리 (파생변수 포함) 및 로드
    # ----------------------------------------------------
    print("=============================================")
    print("단계 1: 데이터 전처리 (파생변수 생성 -> VIF -> 3개 그룹 통합)")
    print("=============================================")
    processor = DataProcessor(data_file, target_col)
    data_sets = processor.load_and_preprocess()

    if not data_sets:
        print("데이터 로드 실패.")
        return

    # ⭐⭐ 핵심 변경: 3개 그룹 통합 데이터셋 사용 (AGGREGATED_SPLIT) ⭐⭐
    # 이제 y값은 0(우량), 1(보통), 2(불량) 만 존재합니다.
    X_train, X_test, y_train, y_test = data_sets['AGGREGATED_SPLIT']

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (변수 중요도 재확인 - 선택 사항)
    # ----------------------------------------------------
    # 3개 그룹에 대해서도 어떤 변수가 유의미한지 p-value를 확인해봅니다.
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (3개 그룹 기준 변수 검증)")
    print("=============================================")

    logreg_trainer = LogRegTrainer()
    _, insignificant_vars = logreg_trainer.train_ordinal_model(X_train, y_train)

    # LogReg에서 유의하지 않다고 나온 변수 제거 (2차 변수 선정)
    if insignificant_vars:
        print(f"\n--> 2차 변수 선정 적용: {len(insignificant_vars)}개 변수 제거 ({insignificant_vars})")
        X_train_final = X_train.drop(columns=insignificant_vars, errors='ignore')
        X_test_final = X_test.drop(columns=insignificant_vars, errors='ignore')
    else:
        print("\n--> 2차 변수 선정에서 추가 제거 변수 없음.")
        X_train_final = X_train
        X_test_final = X_test

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 튜닝 (수동 가중치 적용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 최종 랜덤 포레스트 튜닝 (3개 그룹 + 수동 가중치)")
    print("=============================================")

    # 튜닝 범위 (시간 절약과 성능 사이의 균형)
    rf_param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_features': [0.5, 0.7]
    }

    rf_trainer = RFTrainer()
    best_rf_model, _ = rf_trainer.tune_and_train(X_train_final, y_train, rf_param_grid)

    # 최종 모델 저장
    rf_trainer.save_model(best_rf_model, 'models/final_rf_model_3groups.pkl')

    # ----------------------------------------------------
    # 단계 4: 최종 모델 평가
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 (성공 케이스)")
    print("=============================================")

    # 3개 그룹에 맞는 레이블 이름
    target_names_3groups = ['우량(0)', '보통(1)', '불량(2)']

    evaluator = ModelEvaluator(target_names=target_names_3groups)
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test, model_name="최종 RF (3그룹+파생변수+가중치)")

    print("\n=============================================")
    print("분석 완료. 불량 그룹(2)의 Recall이 0.60~0.80 수준으로 회복되었는지 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()