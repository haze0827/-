# main.py
import pandas as pd
import numpy as np
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
from src.evaluator import ModelEvaluator


def run_analysis():
    # --- 환경 설정 (파일명과 타겟 컬럼명을 정확히 확인하세요) ---
    data_file = 'data/raw/금융데이터셋(Kospi)__673(version 2).xls'
    target_col = 'KIS 신용평점/0A3010'

    # ----------------------------------------------------
    # 단계 1: 데이터 전처리 (EDA, MICE, VIF, Y 분리 및 3그룹 통합)
    # ----------------------------------------------------
    print("=============================================")
    print("단계 1: 데이터 전처리 및 1차 변수 선정 (VIF)")
    print("=============================================")
    processor = DataProcessor(data_file, target_col)
    data_sets = processor.load_and_preprocess()

    if not data_sets:
        print("분석을 계속할 수 없습니다.")
        return

    # LogReg용 데이터 (10개 등급)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = data_sets['ORIGINAL_SPLIT']

    # RF 튜닝 및 최종 평가용 데이터 (3개 그룹)
    X_train_agg, X_test_agg, y_train_agg, y_test_agg = data_sets['AGGREGATED_SPLIT']

    # --- 2차 변수 선정 준비 ---

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (원본 10등급 사용) - ⚠️ 오류로 인해 우회
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (원본 10등급 사용) - ⚠️ 오류로 인해 우회")
    print("=============================================")

    # statsmodels 오류로 인해 LogReg 단계는 건너뛰고, VIF 1차 선정 변수로 최종 RF 튜닝을 진행합니다.
    print("\n!! [조치] LogReg 오류로 인해 2차 변수 선정은 생략하고,")
    print("!! 1차 선정된 VIF 변수로 최종 Random Forest 튜닝을 진행합니다.")

    # 2차 변수 선정 없이, VIF로 1차 선정된 X_train_agg 변수를 최종 변수로 사용합니다.
    X_train_final = X_train_agg
    X_test_final = X_test_agg

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 하이퍼파라미터 튜닝 및 최종 선정
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (3개 그룹 사용)")
    print("=============================================")

    # 튜닝 범위
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_features': [0.1, 0.25]
    }

    rf_trainer = RFTrainer()
    best_rf_model, best_params = rf_trainer.tune_and_train(X_train_final, y_train_agg, param_grid)

    # ... (단계 3: 랜덤 포레스트 튜닝 후) ...
    # 훈련된 모델 저장
    model_save_path = 'models/random_forest_final.pkl'
    rf_trainer.save_model(best_rf_model, model_save_path)

    # ----------------------------------------------------
    # 단계 4: 최종 선정 모델 평가 및 특성 중요도 확인 (OLS 대체 논리)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 및 특성 중요도 확인")
    print("=============================================")

    # 3개 그룹의 레이블 이름 정의
    # 0: 최우량/우량 (저위험), 2: 불량/고위험
    target_names = ['우량(0)', '보통(1)', '불량(2)']

    # [오류 수정] ModelEvaluator 객체를 여기서 생성합니다.
    evaluator = ModelEvaluator(target_names=target_names)

    # 2차 선정된 X_test_final과 3개 그룹 y_test_agg를 사용합니다.
    evaluator.evaluate_model(best_rf_model, X_test_final, y_test_agg, model_name="최종 Random Forest")

    # OLS를 대체하는 핵심 논리: 특성 중요도 추출
    print("\n--- 특성 중요도 (Feature Importance) ---")
    importances = best_rf_model.feature_importances_
    feature_names = X_train_final.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # VIF로 선정된 15개 변수 전체의 중요도를 출력하여 보고서에 활용합니다.
    print("최종 모델의 예측에 가장 큰 영향을 미친 변수 목록 (OLS 대체 근거):")
    print(feature_importance_df)

    print("\n=============================================")
    print("분석 완료. 핵심: Recall(재현율)과 Precision(정밀도) 균형을 확인하세요.")
    print("=============================================")
if __name__ == "__main__":
    run_analysis()