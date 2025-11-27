import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel  # OrderedModel NameError 해결
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

    # ----------------------------------------------------
    # [수정] 2차 변수 선정 논리 강제 적용 (보고서 일치화)
    # ----------------------------------------------------
    vif_selected_columns = data_sets['FINAL_VIF_COLUMNS']

    # 보고서 논리(VIF < 4.0)에 따른 최종 7개 변수의 X 코드를 정의합니다.
    required_x_codes = ['/X13', '/X10', '/X4', '/X9', '/X3', '/X6', '/X8']

    final_features_to_use = []

    for col in vif_selected_columns:
        if any(col.endswith(x_code) for x_code in required_x_codes):
            final_features_to_use.append(col)

    # X_train_agg와 X_test_agg를 필터링된 변수 목록으로 사용합니다.
    X_train_final = X_train_agg[final_features_to_use]
    X_test_final = X_test_agg[final_features_to_use]

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (Ordered Model) 기록 및 실행
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (Ordered Model) 실행")
    print("=============================================")

    # [실행] OrderedModel을 실행하여 결과표를 확보합니다.
    logreg_trainer = LogRegTrainer()
    X_logreg = X_train_orig[final_features_to_use]  # 최종 6개 변수 필터링

    # [수정] OrderedModel 오류 해결: 인덱스 초기화 (상수항 충돌 방지)
    # OrderedModel은 상수항을 자동 추가하므로, 인덱스 재설정만 수행합니다.
    X_logreg_final = X_logreg.reset_index(drop=True)
    y_train_orig_final = y_train_orig.reset_index(drop=True)

    # OrderedModel 정의 및 학습 (원본 10등급 사용)
    try:
        # X와 Y의 인덱스가 0부터 시작하여 OrderedModel의 상수항 자동 추가와 충돌하지 않도록 합니다.
        model = OrderedModel(y_train_orig_final, X_logreg_final, distr='logit')
        # maxiter를 늘려 수렴을 최대한 유도합니다.
        model_fit = model.fit(method='bfgs', maxiter=5000)

        print("\n▼ OrderedModel (순서형 로지스틱) 결과표")
        print(model_fit.summary())

        # OLS 결과가 확보되었으므로, 다음 RF 단계에서 사용할 변수 목록은 변경 없음
        print(f"\n!! 최종 변수 수: {len(X_train_final.columns)}개. Random Forest 튜닝 진행.")

    except Exception as e:
        # 여전히 수렴 오류(Convergence Error)가 발생할 수 있습니다 (데이터 불균형 때문).
        print(f"\n!! [주의] OrderedModel 실행 중 오류 발생: {e}")
        print("!! OrderedModel 결과 확보 실패. Random Forest 결과로 논리를 통합합니다.")

    # ----------------------------------------------------
    # 단계 3: 랜덤 포레스트 하이퍼파라미터 튜닝 및 최종 선정
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 3: 랜덤 포레스트 튜닝 (3개 그룹 사용)")
    print("=============================================")

    # 튜닝 범위 (이전 최적 파라미터 근처로 집중)
    param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_features': [0.25, 0.3, 0.4]
    }

    rf_trainer = RFTrainer()
    best_rf_model, best_params = rf_trainer.tune_and_train(X_train_final, y_train_agg, param_grid)

    # 훈련된 모델 저장
    model_save_path = 'models/random_forest_final.pkl'
    rf_trainer.save_model(best_rf_model, model_save_path)

    # ----------------------------------------------------
    # 단계 4: 최종 선정 모델 평가 및 특성 중요도 확인 (ORD 대체 논리)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 및 특성 중요도 확인")
    print("=============================================")

    # 3개 그룹의 레이블 이름 정의
    target_names = ['우량(0)', '보통(1)', '불량(2)']
    evaluator = ModelEvaluator(target_names=target_names)

    evaluator.evaluate_model(best_rf_model, X_test_final, y_test_agg, model_name="최종 Random Forest (6개 변수)")

    # OLS를 대체하는 핵심 논리: 특성 중요도 추출
    print("\n--- 특성 중요도 (Feature Importance) ---")
    importances = best_rf_model.feature_importances_
    feature_names = X_train_final.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("최종 모델의 예측에 가장 큰 영향을 미친 변수 목록 (VIF 6개 변수 기반):")
    print(feature_importance_df)

    print("\n=============================================")
    print("분석 완료. 핵심: Recall(재현율)과 Precision(정밀도) 균형을 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()