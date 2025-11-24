# src/data_processor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DataProcessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column

    def load_data(self):
        """엑셀 파일 로드 및 초기 EDA 출력"""
        print(f"[1-1] 엑셀 파일 로드 중: {self.file_path}")
        try:
            df = pd.read_excel(self.file_path, sheet_name=0)

            # --- 1. 초기 데이터 탐색 (EDA 출력) ---
            print("\n--- 1.1 데이터 초기 5행 확인 ---")
            print(df.head())
            print("\n--- 1.2 데이터 타입 및 특수문자 포함 여부 확인 (info) ---")
            df.info(verbose=False)
            print("\n--- 1.3 초기 결측치 확인 ---")
            missing_count = df.isnull().sum()
            missing_ratio = (missing_count / len(df)) * 100
            missing_df = pd.DataFrame({'Count': missing_count, 'Ratio': missing_ratio})
            print(missing_df[missing_df['Count'] > 0])
            print("\n--- 1.4 종속변수 분포 확인 (원본 10개 등급) ---")
            print(df[self.target_column].value_counts().sort_index())

            return df
        except FileNotFoundError:
            print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인하세요: {self.file_path}")
            return None

    def apply_log_transform(self, X):
        """특이치 및 왜도 해소를 위한 로그 변환 로직"""
        print("[1-2] 로그 변환 적용 중...")
        X_numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in X_numeric_cols:
            if (X[col].min() >= 0):
                X[col] = np.log1p(X[col])
        return X

    def apply_mice_imputation(self, X):
        """MICE를 이용한 결측치 대체"""
        print("[1-3] MICE 결측치 대체 적용 중...")
        X_numeric = X.select_dtypes(include=[np.number])
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_imputed_array = imputer.fit_transform(X_numeric)
        X_imputed = pd.DataFrame(X_imputed_array, columns=X_numeric.columns, index=X_numeric.index)

        print("MICE 후 결측치 재확인:", X_imputed.isnull().sum().sum())
        return X_imputed

    def remove_multicollinearity(self, X, threshold=10.0):
        """VIF를 이용한 다중공선성 변수 제거 (1차 변수 선정)"""
        print(f"[1-4] VIF 기반 다중공선성 검사 (Threshold={threshold})...")
        df_vif = X.copy()
        removed_vars = []
        while True:
            X_with_const = np.column_stack([np.ones(df_vif.shape[0]), df_vif])

            try:
                vifs = [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
            except np.linalg.LinAlgError:
                print("경고: 행렬이 특이하여 VIF 계산 중단. 변수 중복 또는 선형 종속성 의심.")
                break

            vifs = vifs[1:]

            if not vifs or len(df_vif.columns) == 0:
                break

            max_vif = max(vifs)

            if max_vif > threshold:
                max_vif_index = vifs.index(max_vif)
                col_to_drop = df_vif.columns[max_vif_index]

                # 'VIF 기준으로 변수 선정' 단계 분석용 출력
                print(f"제거 후보: VIF {max_vif:.2f}로 높은 컬럼 '{col_to_drop}'")

                df_vif = df_vif.drop(columns=[col_to_drop])
                removed_vars.append(col_to_drop)
            else:
                break

        print(f"최종 1차 선정 변수 수: {df_vif.shape[1]}")
        return df_vif, removed_vars

    def aggregate_classes(self, y_series):
        """총 10개 등급을 3개 그룹으로 통합 (0=고위험, 2=저위험)"""
        print("[1-5] 신용 등급을 3개 그룹으로 통합 중...")

        # 1-4등급: 고위험 (High Risk) -> 0
        # 5-7등급: 중위험 (Medium Risk) -> 1
        # 8-10등급: 저위험 (Low Risk) -> 2

        def map_grade(grade):
            if grade <= 4:
                return 0
            elif grade <= 7:
                return 1
            else:  # grade 8, 9, 10
                return 2

        y_zero_indexed = y_series.apply(map_grade)

        print("통합된 등급 분포 (0, 1, 2):")
        print(y_zero_indexed.value_counts().sort_index())

        return y_zero_indexed

    def load_and_preprocess(self):
        """모든 전처리 단계를 순서대로 실행하고 두 가지 버전의 Y를 반환"""
        df = self.load_data()
        if df is None:
            return {}

        # 1. 종속변수(y)와 독립변수(X) 분리 및 불필요 컬럼 제거
        y_full = df[self.target_column]
        # KIS, Name, 신용등급(우량,불량) 등 비수치형 컬럼 제거
        X_full = df.drop(columns=[self.target_column, 'KIS', 'Name', '신용등급(우량,불량)'])

        # 2. 전처리 파이프라인
        X_features = self.apply_log_transform(X_full.copy())
        X_features = self.apply_mice_imputation(X_features)

        # 3. VIF 기반 다중공선성 제거 (1차 변수 선정)
        X_vif_selected, _ = self.remove_multicollinearity(X_features)
        y_vif_final = y_full.loc[X_vif_selected.index]

        # 4. Y 버전 준비
        y_aggregated = self.aggregate_classes(y_vif_final)  # 3개 그룹 (0, 1, 2)
        y_original = y_vif_final  # 10개 등급 (LogReg용)

        print("\n전처리 완료. 학습/평가 데이터 분리 시작.")

        # 5. 데이터 분리: LogReg용 (Original Y)
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_vif_selected, y_original, test_size=0.2, random_state=42, stratify=y_original
        )

        # 6. 데이터 분리: RF용 (Aggregated Y)
        X_train_agg, X_test_agg, y_train_agg, y_test_agg = train_test_split(
            X_vif_selected, y_aggregated, test_size=0.2, random_state=42, stratify=y_aggregated
        )

        return {
            'ORIGINAL_SPLIT': (X_train_orig, X_test_orig, y_train_orig, y_test_orig),
            'AGGREGATED_SPLIT': (X_train_agg, X_test_agg, y_train_agg, y_test_agg),
        }