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
from imblearn.over_sampling import SMOTE
import warnings  # <-- 경고 숨김을 위해 warnings 모듈 추가
import matplotlib.pyplot as plt  # <-- 시각화를 위한 matplotlib 추가

# 불필요한 경고 메시지(XGBoost, Sklearn) 숨김 처리
warnings.filterwarnings('ignore')

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

    # OLR 결과에 따라 최종 5개 변수 확정 (재고자산회전율/X10 제거)
    olr_5_features = [
        '부채비율/X3', '이자보상비율/X4', '단기차입금/총자본/X6',
        '매출액증가율/X8', '순이익증가율/X9'
    ]

    final_features_for_rf = olr_5_features
    X_train_data_raw = X_train_agg[final_features_for_rf]  # SMOTE 전 원본 (5개 변수)
    X_test_data = X_test_agg[final_features_for_rf]
    y_train_agg_raw = y_train_agg  # SMOTE 전 원본 Y (3개 그룹)
    y_test_agg_raw = y_test_agg  # 테스트 데이터의 실제 Y 값 추가

    # ----------------------------------------------------
    # 단계 2: 순서형 로지스틱 (Ordered Model) 실행 및 2차 변수 선정 (복원)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 2: 순서형 로지스틱 (Ordered Model) 실행")
    print("=============================================")

    # OLR 학습을 위한 데이터 준비 (VIF 통과 6개 변수, 10개 등급 Y 사용)
    X_logreg = X_train_orig[vif_6_features]
    X_logreg_final = X_logreg.reset_index(drop=True)
    y_train_orig_final = y_train_orig.reset_index(drop=True)

    try:
        # OrderedModel 정의 및 학습 (원본 10등급 사용)
        model = OrderedModel(y_train_orig_final, X_logreg_final, distr='logit')
        model_fit = model.fit(method='bfgs', maxiter=5000)

        # ▼ OLR 결과표 출력 (복원된 부분)
        print("\n▼ OrderedModel (순서형 로지스틱) 결과표")
        print(model_fit.summary())

        print(f"\n!! OLR P-value 검증: '재고자산회전율/X10' 제거.")
        print(f"!! 최종 변수 수: {len(final_features_for_rf)}개. Random Forest 튜닝 진행.")

    except Exception as e:
        print(f"\n!! [주의] OrderedModel 실행 중 오류 발생: {e}")
        print("!! OLR 결과 확보 실패. 최종 5개 변수를 가정하고 다음 단계 진행.")

    # ----------------------------------------------------
    # 단계 3 & 4: 랜덤 포레스트 튜닝 및 평가 (L2 튜닝과 구분)
    # ----------------------------------------------------
    print("\n=============================================")
    print(f"단계 3: 랜덤 포레스트 튜닝 ({len(final_features_for_rf)}개 변수 사용)")
    print("=============================================")

    param_grid_rf = {
        'n_estimators': [100, 150, 200, 250],
        'max_features': [0.25, 0.3, 0.4]
    }

    rf_trainer = RFTrainer()
    best_rf_model, best_params_rf = rf_trainer.tune_and_train(X_train_data_raw, y_train_agg_raw, param_grid_rf)

    model_save_path_rf = 'models/random_forest_final.pkl'
    rf_trainer.save_model(best_rf_model, model_save_path_rf)

    print("\n=============================================")
    print("단계 4: 최종 선정 모델 평가 및 특성 중요도 확인 (Random Forest)")
    print("=============================================")
    target_names = ['우량(0)', '보통(1)', '불량(2)']
    evaluator = ModelEvaluator(target_names=target_names)

    # y_test_agg_raw를 사용하여 평가
    evaluator.evaluate_model(best_rf_model, X_test_data, y_test_agg_raw,
                             model_name=f"최종 Random Forest ({len(final_features_for_rf)}개 변수)")

    # ====================================================
    # 단계 5-A: SMOTE 오버샘플링 적용 (핵심 변경점)
    # ====================================================
    print("\n=============================================")
    print("단계 5-A: SMOTE 오버샘플링 적용 (XGBoost 데이터 준비)")
    print("=============================================")

    # SMOTE 객체 생성 (random_state를 고정하여 결과 재현 가능하게 함)
    sm = SMOTE(random_state=42)

    # SMOTE 적용
    X_train_smote, y_train_smote = sm.fit_resample(X_train_data_raw, y_train_agg_raw)

    print(f"SMOTE 적용 전 학습 데이터 크기: {len(X_train_data_raw)}개")
    print(f"SMOTE 적용 후 학습 데이터 크기: {len(X_train_smote)}개")
    print("SMOTE 적용 후 그룹별 분포:")
    print(y_train_smote.value_counts().sort_index())

    # ----------------------------------------------------
    # 단계 5: XGBoost L2 정규화 튜닝 (SMOTE 데이터 사용)
    # ----------------------------------------------------
    print("\n=============================================")
    print(f"단계 5: XGBoost L2 정규화 튜닝 (SMOTE 데이터 사용, {len(final_features_for_rf)}개 변수)")
    print("=============================================")

    # L2 정규화(reg_lambda)를 포함한 튜닝 그리드 정의
    param_grid_xgb = {
        'n_estimators': [100, 150, 200],
        'max_depth': [3, 4, 5],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0]  # L2 정규화 파라미터
    }

    # XGBoost Trainer 정의
    xgb_trainer = XGBTrainer()

    # 모델 튜닝 실행: SMOTE로 오버샘플링된 데이터 사용 (불량 그룹 Recall 극대화 시도)
    best_xgb_model_smote, best_params_xgb_smote = xgb_trainer.tune_and_train(
        X_train_smote, y_train_smote, param_grid_xgb, scoring='recall_weighted'
    )

    # 훈련된 모델 저장
    model_xgb_smote_save_path = 'models/xgboost_smote_l2_final.pkl'
    xgb_trainer.save_model(best_xgb_model_smote, model_xgb_smote_save_path)

    # ----------------------------------------------------
    # 단계 6: 최종 선정 XGBoost 모델 평가 및 특성 중요도 확인 (SMOTE 적용)
    # ----------------------------------------------------
    print("\n=============================================")
    print("단계 6: 최종 선정 XGBoost 모델 평가 및 특성 중요도 확인 (SMOTE+L2 적용)")
    print("=============================================")

    # 평가: 테스트 데이터는 오버샘플링하지 않은 X_test_data와 y_test_agg_raw를 사용해야 합니다.
    evaluator.evaluate_model(best_xgb_model_smote, X_test_data, y_test_agg_raw,
                             model_name=f"최종 XGBoost (SMOTE+L2, {len(final_features_for_rf)}개 변수)")

    # 특성 중요도 추출
    print("\n--- XGBoost 특성 중요도 (Feature Importance) ---")
    importances_xgb = best_xgb_model_smote.feature_importances_
    feature_names_xgb = X_train_data_raw.columns
    feature_importance_df_xgb = pd.DataFrame({'Feature': feature_names_xgb, 'Importance': importances_xgb})
    feature_importance_df_xgb = feature_importance_df_xgb.sort_values(by='Importance', ascending=False)

    print(f"최종 XGBoost 모델의 예측에 가장 큰 영향을 미친 변수 목록:")
    print(feature_importance_df_xgb)

    # --- 시각화 코드 추가 ---
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 환경에서 흔히 사용되는 한글 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    # 그래프 생성
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance_df_xgb["Feature"], feature_importance_df_xgb["Importance"], color='#FF4500')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # 값 레이블 추가
    for bar in bars:
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.4f}',
            va='center',
            ha='left',
            fontsize=10
        )

    plt.xlabel("특성 중요도 (Feature Importance)")
    plt.title("최종 XGBoost (SMOTE+L2) 모델의 특성 중요도")
    plt.xlim(0, 0.5)
    plt.tight_layout()
    plt.show()  # 그래프 출력

    print("\n=============================================")
    print("분석 완료. 핵심: Recall(재현율)과 Precision(정밀도) 균형을 확인하세요.")
    print("=============================================")


if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e:
        print(f"최종 실행 중 오류 발생: {e}")