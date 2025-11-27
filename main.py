import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from src.data_processor import DataProcessor
from src.logreg_trainer import LogRegTrainer
from src.rf_trainer import RFTrainer
from src.evaluator import ModelEvaluator
from src.xgb_trainer import XGBTrainer
import joblib
# 불균형 해소 알고리즘 (SMOTE) 추가
from imblearn.over_sampling import SMOTE
# 사용자 요청 코드에 필요한 Metric 함수 추가 (Inline 평가를 위함)
from sklearn.metrics import accuracy_score, classification_report


def run_analysis():
    # --- 환경 설정 (파일명과 타겟 컬럼명을 정확히 확인하세요) ---
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

    # 데이터 분리 결과
    X_train_orig, _, y_train_orig, _ = data_sets['ORIGINAL_SPLIT']
    X_train_agg, X_test_agg, y_train_agg, y_test_agg = data_sets['AGGREGATED_SPLIT']

    # VIF를 통과한 6개 변수 (이전 로그 기반)
    vif_6_features = [
        '부채비율/X3',
        '이자보상비율/X4',
        '단기차입금/총자본/X6',
        '매출액증가율/X8',
        '순이익증가율/X9',
        '재고자산회전율/X10'
    ]

    # VIF 6개 변수로 데이터셋 필터링
    X_train_data = X_train_agg[vif_6_features]
    X_test_data = X_test_agg[vif_6_features]

    # OLR 결과에 따라 최종 5개 변수 확정 (재고자산회전율/X10 제거)
    olr_5_features = [
        '부채비율/X3', '이자보상비율/X4', '단기차입금/총자본/X6',
        '매출액증가율/X8', '순이익증가율/X9'
    ]

    final_features_for_rf = olr_5_features
    X_train_data = X_train_agg[final_features_for_rf]
    X_test_data = X_test_agg[final_features_for_rf]

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (Ordered Model) 실행 및 2차 변수 선정
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (Ordered Model) 실행")
    print("=============================================")

    logreg_trainer = LogRegTrainer()
    X_logreg = X_train_orig[vif_6_features]

    X_logreg_final = X_logreg.reset_index(drop=True)
    y_train_orig_final = y_train_orig.reset_index(drop=True)

    try:
        model = OrderedModel(y_train_orig_final, X_logreg_final, distr='logit')
        model_fit = model.fit(method='bfgs', maxiter=5000)

        print("\n▼ OrderedModel (순서형 로지스틱) 결과표")
        print(model_fit.summary())

        print(f"\n!! OLR P-value 검증: '재고자산회전율/X10' 제거.")
        print(f"!! 최종 변수 수: {len(final_features_for_rf)}개. Random Forest 튜닝 진행.")

    except Exception as e:
        print(f"\n!! [주의] OrderedModel 실행 중 오류 발생: {e}")
        print("!! OLR 결과 확보 실패. VIF 6개 변수로 Random Forest 진행합니다.")

    # ----------------------------------------------------
    # 단계 3 & 4: 랜덤 포레스트 튜닝 및 평가
    # (코드 중략: RF 튜닝 및 평가는 이전과 동일하게 5개 변수로 진행)
    # ----------------------------------------------------
    print("\n=============================================")
    print(f"단계 3: 랜덤 포레스트 튜닝 ({len(final_features_for_rf)}개 변수 사용)")
    print("=============================================")

    param_grid = {
        'n_estimators': [100, 150, 200, 250],
        'max_features': [0.25, 0.3, 0.4]
    }

    rf_trainer = RFTrainer()
    best_rf_model, best_params = rf_trainer.tune_and_train(X_train_data, y_train_agg, param_grid)

    model_save_path = 'models/random_forest_final.pkl'
    rf_trainer.save_model(best_rf_model, model_save_path)

    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 및 특성 중요도 확인 (Random Forest)")
    print("=============================================")
    target_names = ['우량(0)', '보통(1)', '불량(2)']
    evaluator = ModelEvaluator(target_names=target_names)
    evaluator.evaluate_model(best_rf_model, X_test_data, y_test_agg,
                             model_name=f"최종 Random Forest ({len(final_features_for_rf)}개 변수)")

    # ----------------------------------------------------
    # 단계 5-A: SMOTE 적용 (데이터 불균형 해소)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 5-A: SMOTE 오버샘플링 적용 (XGBoost용 데이터 준비)")
    print("=============================================")

    # SMOTE 객체 생성 (random_state를 고정하여 결과 재현 가능)
    smote = SMOTE(random_state=42)

    # SMOTE 적용: 불량(2) 그룹의 표본 수를 늘려 균형 잡힌 학습 데이터 생성
    # X_train_data는 5개 변수로 이미 필터링된 데이터입니다.
    X_train_smote, y_train_smote = smote.fit_resample(X_train_data, y_train_agg)

    print(f"SMOTE 적용 전 학습 데이터 크기: {len(X_train_data)}개")
    print(f"SMOTE 적용 후 학습 데이터 크기: {len(X_train_smote)}개")
    print("SMOTE 적용 후 그룹별 분포:")
    print(y_train_smote.value_counts().sort_index())

    # ====================================================
    # 단계 5: XGBoost (SMOTE 데이터 + 가중치 기반 훈련)
    # ====================================================
    print("\n=============================================")
    print(f"단계 5: XGBoost 가중치 기반 훈련 (SMOTE 데이터 사용)")
    print("=============================================")

    # 1. XGBoost Trainer 정의
    xgb_trainer = XGBTrainer(n_estimators=150)

    # 2. 모델 훈련: SMOTE로 오버샘플링된 데이터셋 사용
    # (XGBTrainer 내부에서 compute_sample_weight='balanced'를 적용하여 훈련)
    best_xgb_model = xgb_trainer.train_model(X_train_smote, y_train_smote)

    # 3. 훈련된 모델 저장
    model_xgb_save_path = 'models/xgboost_smote_final.pkl'
    xgb_trainer.save_model(best_xgb_model, model_xgb_save_path)

    # ----------------------------------------------------
    # 단계 6: 최종 선정 XGBoost 모델 평가 및 특성 중요도 확인
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 6: 최종 선정 XGBoost 모델 평가 및 특성 중요도 확인 (SMOTE)")
    print("=============================================")

    # 평가: 테스트 데이터는 원본 데이터를 그대로 사용해야 합니다. (X_test_data)
    evaluator.evaluate_model(best_xgb_model, X_test_data, y_test_agg,
                             model_name=f"최종 XGBoost (SMOTE+가중치, {len(final_features_for_rf)}개 변수)")

    # 특성 중요도 추출
    print("\n--- XGBoost 특성 중요도 (Feature Importance) ---")
    importances_xgb = best_xgb_model.feature_importances_
    feature_names_xgb = X_train_data.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': feature_names_xgb, 'Importance': importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

    print(f"최종 XGBoost 모델의 예측에 가장 큰 영향을 미친 변수 목록:")
    print(feature_importance_df_xgb)

    print("\n=============================================")
    print("분석 완료. 핵심: Recall(재현율)과 Precision(정밀도) 균형을 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    run_analysis()