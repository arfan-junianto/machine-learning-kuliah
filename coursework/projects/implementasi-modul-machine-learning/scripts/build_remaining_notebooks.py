"""Bangun notebook Bab 1–11 dan 13 secara deterministik."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOKS_ROOT = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOK_SPECS = []


def clean(text):
    return dedent(text).strip() + "\n"


def register(chapter, folder, filename, title, pages, cells):
    NOTEBOOK_SPECS.append((chapter, folder, filename, title, pages, cells))


def build_notebook(chapter, folder, filename, title, pages, cells):
    notebook_cells = []
    for index, (kind, source) in enumerate(cells, start=1):
        cell_id = f"bab{chapter:02d}-{index:02d}"
        if kind == "markdown":
            cell = nbf.v4.new_markdown_cell(clean(source), id=cell_id)
        else:
            cell = nbf.v4.new_code_cell(clean(source), id=cell_id)
        notebook_cells.append(cell)

    notebook = nbf.v4.new_notebook(
        cells=notebook_cells,
        metadata={
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
    )
    target = NOTEBOOKS_ROOT / folder / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)
    print(f"generated: {target.relative_to(NOTEBOOKS_ROOT.parent)}")


register(
    1,
    "bab-01-konsep-dasar-machine-learning",
    "implementasi-bab-01-konsep-dasar.ipynb",
    "Konsep Dasar Machine Learning",
    "14–15",
    [
        (
            "markdown",
            """
            # Implementasi Bab 1 — Konsep Dasar Machine Learning

            Alur pertama machine learning: memuat data, eksplorasi, split, training KNN, evaluasi, dan visualisasi Iris [ML-MODUL-01, hlm. 14–15]. Notebook ini merapikan baris kode yang terpotong pada PDF dan menambahkan eksperimen nilai k sebagai latihan.
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_iris
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.neighbors import KNeighborsClassifier

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            print("Library berhasil dimuat.")
            """,
        ),
        (
            "code",
            """
            iris = load_iris(as_frame=True)
            df = iris.frame.copy()
            df["target_name"] = df["target"].map(dict(enumerate(iris.target_names)))
            print("Ukuran dataset:", df.shape)
            display(df.head())
            """,
        ),
        (
            "code",
            """
            print("Missing value:")
            print(df.isna().sum())
            print("\\nJumlah data per kelas:")
            print(df["target_name"].value_counts())

            X = df[iris.feature_names]
            y = df["target"]
            """,
        ),
        (
            "code",
            """
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_train, y_train)
            print("Model dilatih pada", X_train.shape[0], "data.")
            """,
        ),
        (
            "code",
            """
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Akurasi test: {accuracy:.2%}")
            print(classification_report(y_test, y_pred, target_names=iris.target_names))
            """,
        ),
        (
            "code",
            """
            plt.figure(figsize=(7, 5))
            for target_value, target_name in enumerate(iris.target_names):
                subset = df[df["target"] == target_value]
                plt.scatter(
                    subset["petal length (cm)"],
                    subset["petal width (cm)"],
                    label=target_name,
                    alpha=0.75,
                )
            plt.xlabel("Panjang Petal (cm)")
            plt.ylabel("Lebar Petal (cm)")
            plt.title("Sebaran Iris Berdasarkan Fitur Petal")
            plt.legend()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan mandiri: pengaruh nilai k

            K kecil cenderung sensitif terhadap noise, sedangkan k terlalu besar dapat mengaburkan batas kelas. Eksperimen berikut merupakan pelengkap untuk menghubungkan pengantar Bab 1 dengan pembahasan KNN pada Bab 6.
            """,
        ),
        (
            "code",
            """
            k_values = range(1, 16)
            k_cv_scores = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for k in k_values:
                candidate = KNeighborsClassifier(n_neighbors=k)
                k_cv_scores.append(cross_val_score(candidate, X_train, y_train, cv=cv).mean())

            best_k = list(k_values)[int(np.argmax(k_cv_scores))]
            selected_knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
            final_test_accuracy = accuracy_score(y_test, selected_knn.predict(X_test))
            print(f"k terbaik menurut CV training: {best_k}")
            print(f"Akurasi test final: {final_test_accuracy:.3f}")
            plt.plot(list(k_values), k_cv_scores, marker="o")
            plt.xlabel("k")
            plt.ylabel("Rata-rata akurasi CV")
            plt.title("Pengaruh Nilai k")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Notebook menunjukkan pola kerja end-to-end paling sederhana: data → eksplorasi → split → training → evaluasi. Akurasi tinggi pada Iris masuk akal karena kelas relatif terpisah pada fitur petal [ML-MODUL-01, hlm. 15].
            """,
        ),
    ],
)


register(
    2,
    "bab-02-workflow-crisp-dm-bias-variance",
    "implementasi-bab-02-workflow-bias-variance.ipynb",
    "Workflow, CRISP-DM, dan Bias-Variance",
    "20–21",
    [
        (
            "markdown",
            """
            # Implementasi Bab 2 — Workflow, CRISP-DM, dan Bias-Variance

            Notebook membuat learning curve untuk mendiagnosis underfitting, seimbang, dan overfitting serta memetakan eksperimen ke enam fase CRISP-DM [ML-MODUL-01, hlm. 20–21].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import make_classification
            from sklearn.model_selection import learning_curve, train_test_split
            from sklearn.tree import DecisionTreeClassifier

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            X, y = make_classification(
                n_samples=1000,
                n_features=10,
                n_informative=6,
                weights=[0.8, 0.2],
                random_state=SEED,
            )
            print("Distribusi kelas:", np.bincount(y))
            """,
        ),
        (
            "code",
            """
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )
            print("Proporsi train:", np.bincount(y_train) / len(y_train))
            print("Proporsi test :", np.bincount(y_test) / len(y_test))
            """,
        ),
        (
            "code",
            """
            model = DecisionTreeClassifier(max_depth=4, random_state=SEED)
            train_sizes, train_scores, validation_scores = learning_curve(
                model,
                X_train,
                y_train,
                cv=5,
                train_sizes=np.linspace(0.1, 1.0, 8),
                scoring="accuracy",
                n_jobs=-1,
            )
            train_mean = train_scores.mean(axis=1)
            validation_mean = validation_scores.mean(axis=1)
            print("Train terbesar:", round(train_mean[-1], 3))
            print("Validasi terbesar:", round(validation_mean[-1], 3))
            print("Gap:", round(train_mean[-1] - validation_mean[-1], 3))
            """,
        ),
        (
            "code",
            """
            plt.plot(train_sizes, train_mean, "o-", label="Training Score")
            plt.plot(train_sizes, validation_mean, "o-", label="Validation Score")
            plt.xlabel("Jumlah Data Latih")
            plt.ylabel("Akurasi")
            plt.title("Learning Curve — max_depth=4")
            plt.legend()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: pemetaan CRISP-DM

            Contoh studi kasus churn pelanggan:

            1. **Business Understanding:** tentukan biaya pelanggan pergi dan target retensi.
            2. **Data Understanding:** audit distribusi churn, kualitas, dan potensi bias.
            3. **Data Preparation:** split, imputasi, encoding, dan pencegahan leakage.
            4. **Modeling:** bandingkan baseline dan beberapa tingkat kompleksitas.
            5. **Evaluation:** gunakan recall/F1 dan validasi lintas fold.
            6. **Deployment:** integrasikan skor risiko dan monitor drift.

            Pemetaan ini adalah jawaban contoh untuk latihan konseptual [ML-MODUL-01, hlm. 21].
            """,
        ),
        (
            "code",
            """
            depth_results = []
            for depth in [1, 4, None]:
                candidate = DecisionTreeClassifier(max_depth=depth, random_state=SEED)
                _, train_scores_i, validation_scores_i = learning_curve(
                    candidate,
                    X_train,
                    y_train,
                    cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 6),
                    scoring="accuracy",
                    n_jobs=-1,
                )
                train_final = train_scores_i.mean(axis=1)[-1]
                validation_final = validation_scores_i.mean(axis=1)[-1]
                gap = train_final - validation_final
                diagnosis = "overfitting" if gap > 0.08 else ("underfitting" if validation_final < 0.75 else "seimbang")
                depth_results.append((str(depth), train_final, validation_final, diagnosis))

            display(pd.DataFrame(depth_results, columns=["max_depth", "train", "validation", "diagnosis"]))
            """,
        ),
        (
            "markdown",
            """
            ## Interpretasi

            Gap train–validation yang kecil pada skor cukup tinggi menunjukkan keseimbangan bias-variance yang lebih baik. Diagnosis tetap perlu dilihat bersama konteks biaya kesalahan dan test set yang tidak ikut memilih model [ML-MODUL-01, hlm. 21].
            """,
        ),
    ],
)


register(
    3,
    "bab-03-data-preparation-lanjutan",
    "implementasi-bab-03-data-preparation.ipynb",
    "Data Preparation Lanjutan",
    "26–27",
    [
        (
            "markdown",
            """
            # Implementasi Bab 3 — Missing Value, Outlier, dan Imbalanced Data

            Implementasi KNNImputer, IsolationForest, SMOTE, undersampling, dan class weight pada data sintetis tidak seimbang [ML-MODUL-01, hlm. 26–27]. Adaptasi penting: split dilakukan sebelum fitting imputasi/outlier/SMOTE agar test set tidak bocor.
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from sklearn.datasets import make_classification
            from sklearn.ensemble import IsolationForest
            from sklearn.impute import KNNImputer, SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score
            from sklearn.model_selection import train_test_split

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            X_array, y = make_classification(
                n_samples=2000,
                n_features=8,
                n_informative=5,
                weights=[0.95, 0.05],
                random_state=SEED,
            )
            feature_names = [f"fitur_{index}" for index in range(8)]
            X = pd.DataFrame(X_array, columns=feature_names)
            rng = np.random.default_rng(SEED)
            X = X.mask(rng.random(X.shape) < 0.05)
            print("Missing per fitur:\\n", X.isna().sum())
            """,
        ),
        (
            "code",
            """
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=SEED
            )

            imputer = KNNImputer(n_neighbors=5)
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test_raw), columns=feature_names, index=X_test_raw.index
            )

            detector = IsolationForest(contamination=0.02, random_state=SEED)
            inlier_mask = detector.fit_predict(X_train) == 1
            X_train_clean = X_train.loc[inlier_mask]
            y_train_clean = pd.Series(y_train, index=X_train_raw.index).loc[inlier_mask]
            print("Outlier train dibuang:", int((~inlier_mask).sum()))
            """,
        ),
        (
            "code",
            """
            baseline = LogisticRegression(max_iter=1000, random_state=SEED)
            baseline.fit(X_train_clean, y_train_clean)

            smote = SMOTE(random_state=SEED)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_clean, y_train_clean)
            model_smote = LogisticRegression(max_iter=1000, random_state=SEED)
            model_smote.fit(X_train_smote, y_train_smote)

            for name, candidate in [("Baseline", baseline), ("SMOTE", model_smote)]:
                prediction = candidate.predict(X_test)
                print(name, "accuracy=", round(accuracy_score(y_test, prediction), 3),
                      "F1=", round(f1_score(y_test, prediction), 3))
            """,
        ),
        (
            "code",
            """
            before = pd.Series(y_train_clean).value_counts().sort_index()
            after = pd.Series(y_train_smote).value_counts().sort_index()
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].bar(before.index.astype(str), before.values)
            axes[0].set_title("Sebelum SMOTE")
            axes[1].bar(after.index.astype(str), after.values)
            axes[1].set_title("Setelah SMOTE")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: mean imputation vs KNN imputation

            Perbandingan dilakukan dengan fitting imputer hanya pada train set. F1 dipakai karena kelas positif langka; akurasi saja dapat menyesatkan [ML-MODUL-01, hlm. 27].
            """,
        ),
        (
            "code",
            """
            imputation_scores = {}
            for name, candidate_imputer in {
                "Mean": SimpleImputer(strategy="mean"),
                "KNN": KNNImputer(n_neighbors=5),
            }.items():
                train_i = candidate_imputer.fit_transform(X_train_raw)
                test_i = candidate_imputer.transform(X_test_raw)
                candidate = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
                candidate.fit(train_i, y_train)
                imputation_scores[name] = f1_score(y_test, candidate.predict(test_i))
            print(imputation_scores)
            """,
        ),
        (
            "code",
            """
            sampler = RandomUnderSampler(random_state=SEED)
            X_train_under, y_train_under = sampler.fit_resample(X_train_clean, y_train_clean)
            strategies = {
                "SMOTE": model_smote,
                "Undersampling": LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_under, y_train_under),
                "Class weight": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X_train_clean, y_train_clean),
            }
            strategy_scores = {
                name: f1_score(y_test, candidate.predict(X_test))
                for name, candidate in strategies.items()
            }
            display(pd.Series(strategy_scores, name="F1").sort_values(ascending=False).to_frame())
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Strategi terbaik tidak selalu yang memberi akurasi tertinggi. Bandingkan F1/recall, jaga semua transformasi tetap belajar dari train set, dan pilih penanganan imbalance sesuai biaya kesalahan kasus nyata.
            """,
        ),
    ],
)


register(
    4,
    "bab-04-feature-engineering-pipeline",
    "implementasi-bab-04-feature-engineering.ipynb",
    "Feature Engineering dan Pipeline",
    "32–34",
    [
        (
            "markdown",
            """
            # Implementasi Bab 4 — Feature Engineering dan Pipeline

            Feature construction, ColumnTransformer, encoding, feature selection, dan Pipeline end-to-end [ML-MODUL-01, hlm. 32–34]. Dataset properti dibuat sintetis agar notebook mandiri di Colab.
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.compose import ColumnTransformer
            from sklearn.datasets import load_breast_cancer
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import RFE, SelectKBest, f_classif
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, r2_score
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            rng = np.random.default_rng(SEED)
            n_samples = 500
            properties = pd.DataFrame({
                "luas_bangunan": rng.normal(120, 40, n_samples).clip(30),
                "jumlah_kamar": rng.integers(1, 6, n_samples),
                "tipe_properti": rng.choice(["Rumah", "Apartemen", "Ruko"], n_samples),
                "kondisi_bangunan": rng.choice(["Buruk", "Sedang", "Baik", "Sangat Baik"], n_samples),
            })
            properties["harga"] = (
                properties["luas_bangunan"] * 15
                + properties["jumlah_kamar"] * 20
                + rng.normal(0, 50, n_samples)
            ) * 1_000_000
            properties["rasio_kamar_per_luas"] = properties["jumlah_kamar"] / properties["luas_bangunan"]
            display(properties.head())
            """,
        ),
        (
            "code",
            """
            numeric_features = ["luas_bangunan", "jumlah_kamar", "rasio_kamar_per_luas"]
            categorical_features = ["tipe_properti"]
            ordinal_features = ["kondisi_bangunan"]

            preprocessor = ColumnTransformer([
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric_features),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), categorical_features),
                ("ord", OrdinalEncoder(categories=[["Buruk", "Sedang", "Baik", "Sangat Baik"]]), ordinal_features),
            ])
            """,
        ),
        (
            "code",
            """
            X = properties.drop(columns="harga")
            y = properties["harga"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED
            )
            property_pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
            ])
            property_pipeline.fit(X_train, y_train)
            y_pred = property_pipeline.predict(X_test)
            print("R2:", round(r2_score(y_test, y_pred), 3))
            """,
        ),
        (
            "code",
            """
            plt.scatter(y_test, y_pred, alpha=0.6)
            limits = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            plt.plot(limits, limits, "r--")
            plt.xlabel("Harga Aktual")
            plt.ylabel("Harga Prediksi")
            plt.title("Prediksi vs Aktual — Pipeline Lengkap")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan: tiga pendekatan feature selection

            Contoh berikut menggunakan dataset klasifikasi bawaan, sehingga tidak memerlukan unduhan Titanic. Filter diwakili SelectKBest, wrapper oleh RFE, dan embedded oleh koefisien Logistic Regression berpenalti L1 [ML-MODUL-01, hlm. 34].
            """,
        ),
        (
            "code",
            """
            cancer = load_breast_cancer(as_frame=True)
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                cancer.data, cancer.target, test_size=0.2, random_state=SEED, stratify=cancer.target
            )

            filter_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=10)),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ])
            filter_pipeline.fit(X_train_c, y_train_c)

            wrapper_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("selector", RFE(LogisticRegression(max_iter=2000, random_state=SEED), n_features_to_select=10)),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ])
            wrapper_pipeline.fit(X_train_c, y_train_c)

            embedded_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000, random_state=SEED)),
            ])
            embedded_pipeline.fit(X_train_c, y_train_c)

            filter_features = list(cancer.feature_names[filter_pipeline.named_steps["selector"].get_support()])
            rfe_features = list(cancer.feature_names[wrapper_pipeline.named_steps["selector"].get_support()])
            embedded_mask = np.abs(embedded_pipeline.named_steps["model"].coef_).max(axis=0) > 1e-8
            embedded_features = list(cancer.feature_names[embedded_mask])
            selected_feature_names = {
                "SelectKBest": filter_features,
                "RFE": rfe_features,
                "L1 embedded": embedded_features,
            }

            for name, candidate in {
                "SelectKBest": filter_pipeline,
                "RFE": wrapper_pipeline,
                "L1 embedded": embedded_pipeline,
            }.items():
                print(name, round(accuracy_score(y_test_c, candidate.predict(X_test_c)), 3))
                print("Fitur terpilih:", selected_feature_names[name])
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 2: pipeline lengkap pada Titanic

            Sesuai instruksi modul, latihan ini memakai fitur numerik `age` dan `fare`, fitur kategorikal `sex` dan `embarked`, lalu membungkus imputasi, scaling, one-hot encoding, dan Logistic Regression dalam satu Pipeline [ML-MODUL-01, hlm. 34]. Jika dataset seaborn tidak dapat diunduh, notebook memakai data fallback dengan skema yang sama agar tetap dapat diuji secara offline.
            """,
        ),
        (
            "code",
            """
            try:
                import seaborn as sns

                titanic = sns.load_dataset("titanic")[
                    ["age", "fare", "sex", "embarked", "survived"]
                ].copy()
                titanic_source = "seaborn Titanic"
            except Exception:
                titanic_rng = np.random.default_rng(SEED)
                fallback_size = 160
                fallback_sex = titanic_rng.choice(["female", "male"], fallback_size)
                fallback_embarked = titanic_rng.choice(["C", "Q", "S"], fallback_size)
                fallback_age = titanic_rng.normal(30, 14, fallback_size).clip(1, 80)
                fallback_fare = titanic_rng.lognormal(3.0, 0.8, fallback_size).clip(4, 300)
                survival_probability = (
                    0.15
                    + 0.50 * (fallback_sex == "female")
                    + 0.15 * (fallback_age < 16)
                    + 0.10 * (fallback_fare > np.median(fallback_fare))
                ).clip(0.05, 0.95)
                titanic = pd.DataFrame({
                    "age": fallback_age,
                    "fare": fallback_fare,
                    "sex": fallback_sex,
                    "embarked": fallback_embarked,
                    "survived": titanic_rng.binomial(1, survival_probability),
                })
                titanic.loc[::13, "age"] = np.nan
                titanic.loc[::17, "embarked"] = np.nan
                titanic_source = "fallback offline berskema Titanic"

            titanic = titanic.dropna(subset=["survived"]).copy()
            print("Sumber data:", titanic_source, "| ukuran:", titanic.shape)
            display(titanic.head())
            """,
        ),
        (
            "code",
            """
            titanic_numeric_features = ["age", "fare"]
            titanic_categorical_features = ["sex", "embarked"]
            titanic_features = titanic_numeric_features + titanic_categorical_features

            titanic_preprocessor = ColumnTransformer([
                ("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), titanic_numeric_features),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), titanic_categorical_features),
            ])
            titanic_pipeline = Pipeline([
                ("preprocessing", titanic_preprocessor),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ])

            X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test = train_test_split(
                titanic[titanic_features],
                titanic["survived"].astype(int),
                test_size=0.2,
                random_state=SEED,
                stratify=titanic["survived"].astype(int),
            )
            titanic_pipeline.fit(X_titanic_train, y_titanic_train)
            titanic_prediction = titanic_pipeline.predict(X_titanic_test)
            titanic_test_accuracy = accuracy_score(y_titanic_test, titanic_prediction)
            print(f"Akurasi test Titanic: {titanic_test_accuracy:.3f}")
            """,
        ),
        (
            "code",
            """
            new_property = pd.DataFrame([{
                "luas_bangunan": 100,
                "jumlah_kamar": 3,
                "tipe_properti": "Rumah",
                "kondisi_bangunan": "Baik",
                "rasio_kamar_per_luas": 3 / 100,
            }])
            print("Prediksi harga data baru:", f"Rp{property_pipeline.predict(new_property)[0]:,.0f}")
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Pipeline menyatukan preprocessing dan model sehingga fitting pada train serta prediksi data baru konsisten. Ini mengurangi risiko leakage dan kesalahan urutan transformasi [ML-MODUL-01, hlm. 34].
            """,
        ),
    ],
)


register(
    5,
    "bab-05-decision-tree-pruning",
    "implementasi-bab-05-decision-tree.ipynb",
    "Decision Tree dan Pruning",
    "39–40",
    [
        (
            "markdown",
            """
            # Implementasi Bab 5 — Decision Tree dan Pruning

            Perbandingan pohon penuh, pre-pruning, dan cost-complexity post-pruning pada Breast Cancer Wisconsin [ML-MODUL-01, hlm. 39–40]. Pemilihan alpha latihan memakai validation set, bukan test set, sebagai adaptasi anti-leakage.
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split
            from sklearn.tree import DecisionTreeClassifier, plot_tree

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            data = load_breast_cancer(as_frame=True)
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full
            )
            print("train/val/test:", X_train.shape, X_val.shape, X_test.shape)
            """,
        ),
        (
            "code",
            """
            tree_full = DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train)
            tree_pruned = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=5, random_state=SEED
            ).fit(X_train, y_train)

            pruning_path = tree_full.cost_complexity_pruning_path(X_train, y_train)
            candidate_alphas = pruning_path.ccp_alphas[:-1]
            validation_scores = []
            for alpha in candidate_alphas:
                candidate = DecisionTreeClassifier(ccp_alpha=alpha, random_state=SEED)
                candidate.fit(X_train, y_train)
                validation_scores.append(accuracy_score(y_val, candidate.predict(X_val)))
            selected_alpha = candidate_alphas[int(np.argmax(validation_scores))]
            tree_ccp = DecisionTreeClassifier(ccp_alpha=selected_alpha, random_state=SEED)
            tree_ccp.fit(X_train_full, y_train_full)
            print("ccp_alpha terpilih:", selected_alpha)
            """,
        ),
        (
            "code",
            """
            comparison = []
            for name, candidate in {
                "Penuh": tree_full,
                "Pre-pruning": tree_pruned,
                "Post-pruning": tree_ccp,
            }.items():
                train_X = X_train_full if name == "Post-pruning" else X_train
                train_y = y_train_full if name == "Post-pruning" else y_train
                train_accuracy = accuracy_score(train_y, candidate.predict(train_X))
                test_accuracy = accuracy_score(y_test, candidate.predict(X_test))
                comparison.append((name, train_accuracy, test_accuracy, train_accuracy - test_accuracy))
            display(pd.DataFrame(comparison, columns=["model", "train", "test", "gap"]))
            """,
        ),
        (
            "code",
            """
            plt.figure(figsize=(16, 8))
            plot_tree(
                tree_pruned,
                feature_names=data.feature_names,
                class_names=data.target_names,
                filled=True,
                rounded=True,
                fontsize=7,
            )
            plt.title("Decision Tree Pre-Pruning")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: kombinasi pre-pruning

            Grid kecil berikut memperlihatkan pengaruh `max_depth` dan `min_samples_leaf` [ML-MODUL-01, hlm. 40]. Kombinasi dipilih memakai validation set, lalu test set hanya dibuka sekali untuk evaluasi final.
            """,
        ),
        (
            "code",
            """
            pre_pruning_results = []
            for depth in [2, 4, 6, None]:
                for min_leaf in [1, 5, 10]:
                    candidate = DecisionTreeClassifier(
                        max_depth=depth, min_samples_leaf=min_leaf, random_state=SEED
                    ).fit(X_train, y_train)
                    validation_accuracy = accuracy_score(y_val, candidate.predict(X_val))
                    pre_pruning_results.append((depth, min_leaf, validation_accuracy))
            pre_pruning_frame = pd.DataFrame(
                pre_pruning_results, columns=["max_depth", "min_leaf", "validation_accuracy"]
            )
            display(pre_pruning_frame.sort_values("validation_accuracy", ascending=False))
            best_row = pre_pruning_frame.sort_values("validation_accuracy", ascending=False).iloc[0]
            selected_pre_pruned_tree = DecisionTreeClassifier(
                max_depth=None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
                min_samples_leaf=int(best_row["min_leaf"]),
                random_state=SEED,
            ).fit(X_train_full, y_train_full)
            final_test_accuracy = accuracy_score(y_test, selected_pre_pruned_tree.predict(X_test))
            print(f"Akurasi test final: {final_test_accuracy:.3f}")
            """,
        ),
        (
            "code",
            """
            plt.plot(candidate_alphas, validation_scores, marker="o")
            plt.axvline(selected_alpha, color="red", linestyle="--", label="alpha terpilih")
            plt.xlabel("ccp_alpha")
            plt.ylabel("Akurasi validation")
            plt.title("Cost Complexity Pruning Path")
            plt.legend()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Pohon penuh mudah menghafal training set. Pruning menukar sedikit akurasi training untuk gap generalisasi yang lebih kecil; parameter sebaiknya dipilih dari validation/CV, bukan test set [ML-MODUL-01, hlm. 40].
            """,
        ),
    ],
)


register(
    6,
    "bab-06-knn-naive-bayes",
    "implementasi-bab-06-knn-naive-bayes.ipynb",
    "KNN dan Naive Bayes",
    "46–47",
    [
        (
            "markdown",
            """
            # Implementasi Bab 6 — KNN dan Naive Bayes

            Perbandingan KNN berskala dengan Gaussian Naive Bayes pada Wine, pengaruh nilai k, dan Multinomial Naive Bayes untuk teks [ML-MODUL-01, hlm. 46–47].
            """,
        ),
        (
            "code",
            """
            import random
            import time
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer, load_wine
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.naive_bayes import GaussianNB, MultinomialNB
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            wine = load_wine(as_frame=True)
            X_train, X_test, y_train, y_test = train_test_split(
                wine.data, wine.target, test_size=0.2, random_state=SEED, stratify=wine.target
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            """,
        ),
        (
            "code",
            """
            knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
            knn.fit(X_train_scaled, y_train)
            gaussian_nb = GaussianNB().fit(X_train, y_train)

            accuracy_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
            accuracy_nb = accuracy_score(y_test, gaussian_nb.predict(X_test))
            print(f"KNN: {accuracy_knn:.3f}")
            print(f"GaussianNB: {accuracy_nb:.3f}")
            """,
        ),
        (
            "code",
            """
            k_values = range(1, 21)
            k_cv_scores = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for k in k_values:
                candidate = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=k)),
                ])
                k_cv_scores.append(cross_val_score(candidate, X_train, y_train, cv=cv).mean())
            best_k = list(k_values)[int(np.argmax(k_cv_scores))]
            selected_knn = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=best_k)),
            ]).fit(X_train, y_train)
            print("k terbaik menurut CV training:", best_k)
            print("Akurasi test final:", round(accuracy_score(y_test, selected_knn.predict(X_test)), 3))
            plt.plot(list(k_values), k_cv_scores, marker="o")
            plt.xlabel("Nilai k")
            plt.ylabel("Rata-rata akurasi CV")
            plt.title("Pengaruh Nilai k")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: waktu KNN vs GaussianNB

            Waktu diukur secara sederhana untuk tujuan pembelajaran. Hasil dipengaruhi perangkat dan ukuran data [ML-MODUL-01, hlm. 47].
            """,
        ),
        (
            "code",
            """
            cancer = load_breast_cancer(as_frame=True)
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                cancer.data, cancer.target, test_size=0.2, random_state=SEED, stratify=cancer.target
            )
            candidates = {
                "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
                "GaussianNB": GaussianNB(),
            }
            timing_rows = []
            for name, candidate in candidates.items():
                start = time.perf_counter()
                candidate.fit(X_train_c, y_train_c)
                fit_time = time.perf_counter() - start
                start = time.perf_counter()
                prediction = candidate.predict(X_test_c)
                predict_time = time.perf_counter() - start
                timing_rows.append((name, accuracy_score(y_test_c, prediction), fit_time, predict_time))
            display(pd.DataFrame(timing_rows, columns=["model", "accuracy", "fit_seconds", "predict_seconds"]))
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 2: Multinomial Naive Bayes untuk teks

            CountVectorizer mengubah kalimat menjadi frekuensi kata non-negatif yang cocok untuk MultinomialNB [ML-MODUL-01, hlm. 47].
            """,
        ),
        (
            "code",
            """
            texts = [
                "dapatkan diskon gratis sekarang",
                "rapat tim jam 10 pagi",
                "menangkan hadiah gratis klik disini",
                "laporan keuangan bulan ini",
                "diskon spesial hari ini",
                "jadwal kuliah minggu depan",
            ]
            labels = [1, 0, 1, 0, 1, 0]
            text_pipeline = Pipeline([
                ("vectorizer", CountVectorizer()),
                ("model", MultinomialNB()),
            ])
            text_pipeline.fit(texts, labels)
            examples = ["hadiah gratis untuk anda", "jadwal rapat besok"]
            print(list(zip(examples, text_pipeline.predict(examples))))
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            KNN sensitif pada skala dan nilai k; GaussianNB memodelkan distribusi fitur kontinu; MultinomialNB cocok untuk hitungan seperti bag-of-words. Pemilihan model harus mengikuti karakteristik data.
            """,
        ),
    ],
)


register(
    7,
    "bab-07-evaluasi-model",
    "implementasi-bab-07-evaluasi-model.ipynb",
    "Evaluasi Model",
    "53–54",
    [
        (
            "markdown",
            """
            # Implementasi Bab 7 — Evaluasi Model Klasifikasi

            Confusion matrix, precision, recall, F1, ROC-AUC, PR curve, dan Stratified k-Fold CV pada data tidak seimbang [ML-MODUL-01, hlm. 53–54].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer, make_classification
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay,
                auc,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_recall_curve,
                precision_score,
                recall_score,
                roc_curve,
            )
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.tree import DecisionTreeClassifier

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            X, y = make_classification(
                n_samples=1000, n_features=10, weights=[0.7, 0.3], random_state=SEED
            )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=SEED, stratify=y
            )
            model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1]
            """,
        ),
        (
            "code",
            """
            matrix = confusion_matrix(y_test, y_pred)
            print("Confusion matrix:\\n", matrix)
            print("Precision:", round(precision_score(y_test, y_pred), 3))
            print("Recall:", round(recall_score(y_test, y_pred), 3))
            print("F1:", round(f1_score(y_test, y_pred), 3))
            print(classification_report(y_test, y_pred))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
            plt.show()
            """,
        ),
        (
            "code",
            """
            fpr, tpr, _ = roc_curve(y_test, y_score)
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            pr_auc = auc(recall, precision)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
            axes[0].legend()
            axes[1].plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
            axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
            axes[1].legend()
            plt.show()
            """,
        ),
        (
            "code",
            """
            stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = cross_val_score(model, X, y, cv=stratified_cv, scoring="f1", n_jobs=-1)
            print("F1 per fold:", np.round(cv_scores, 3))
            print(f"Mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: laporan dua model pada data medis

            Dataset Breast Cancer dipakai sebagai latihan akademik, bukan alat diagnosis. Logistic Regression dibungkus scaling dalam Pipeline agar CV bebas leakage [ML-MODUL-01, hlm. 54].
            """,
        ),
        (
            "code",
            """
            cancer = load_breast_cancer()
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                cancer.data, cancer.target, test_size=0.2, random_state=SEED, stratify=cancer.target
            )
            report_models = {
                "Logistic Regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
                ]),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
            }
            rows = []
            exercise_fig, exercise_axes = plt.subplots(2, 3, figsize=(15, 8))
            for row_index, (name, candidate) in enumerate(report_models.items()):
                candidate.fit(X_train_c, y_train_c)
                prediction = candidate.predict(X_test_c)
                probability = candidate.predict_proba(X_test_c)[:, 1]
                rows.append((
                    name,
                    precision_score(y_test_c, prediction),
                    recall_score(y_test_c, prediction),
                    f1_score(y_test_c, prediction),
                ))
                ConfusionMatrixDisplay.from_predictions(
                    y_test_c, prediction, cmap="Blues", ax=exercise_axes[row_index, 0]
                )
                exercise_axes[row_index, 0].set_title(f"Confusion Matrix — {name}")
                exercise_roc = roc_curve(y_test_c, probability)
                exercise_axes[row_index, 1].plot(exercise_roc[0], exercise_roc[1])
                exercise_axes[row_index, 1].plot([0, 1], [0, 1], "--", color="gray")
                exercise_axes[row_index, 1].set_title(f"ROC — {name}")
                exercise_pr = precision_recall_curve(y_test_c, probability)
                exercise_axes[row_index, 2].plot(exercise_pr[1], exercise_pr[0])
                exercise_axes[row_index, 2].set_title(f"Precision-Recall — {name}")
            plt.tight_layout()
            plt.show()
            display(pd.DataFrame(rows, columns=["model", "precision", "recall", "f1"]))
            """,
        ),
        (
            "code",
            """
            model_list = {
                "Logistic Regression": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
                ]),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
                "KNN": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=5)),
                ]),
            }
            comparison = []
            for name, candidate in model_list.items():
                scores = cross_val_score(candidate, X, y, cv=stratified_cv, scoring="f1", n_jobs=-1)
                comparison.append((name, scores.mean(), scores.std()))
            display(pd.DataFrame(comparison, columns=["model", "mean_f1", "std_f1"]).sort_values("mean_f1", ascending=False))
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            ROC-AUC mengukur ranking lintas threshold, sedangkan PR curve lebih informatif saat kelas positif langka. Mean dan standar deviasi CV perlu dibaca bersama untuk menilai performa sekaligus stabilitas [ML-MODUL-01, hlm. 54].
            """,
        ),
    ],
)


register(
    8,
    "bab-08-support-vector-machine",
    "implementasi-bab-08-svm.ipynb",
    "Support Vector Machine",
    "58–59",
    [
        (
            "markdown",
            """
            # Implementasi Bab 8 — Support Vector Machine

            Scaling, GridSearchCV, evaluasi, heatmap parameter, perbandingan kernel, dan decision boundary dua fitur [ML-MODUL-01, hlm. 58–59].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_breast_cancer
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            data = load_breast_cancer(as_frame=True)
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
            )
            svm_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC()),
            ])
            """,
        ),
        (
            "code",
            """
            parameter_grid = {
                "svm__C": [0.1, 1, 10],
                "svm__gamma": [0.001, 0.01, 0.1],
                "svm__kernel": ["rbf", "linear"],
            }
            grid_search = GridSearchCV(
                svm_pipeline, parameter_grid, cv=5, scoring="f1", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            print("Parameter terbaik:", grid_search.best_params_)
            print("F1 CV terbaik:", round(grid_search.best_score_, 3))
            """,
        ),
        (
            "code",
            """
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            decision_scores = best_model.decision_function(X_test)
            print("Akurasi test:", round(accuracy_score(y_test, y_pred), 3))
            print("Rentang decision_function:", decision_scores.min(), decision_scores.max())
            print(classification_report(y_test, y_pred, target_names=data.target_names))
            """,
        ),
        (
            "code",
            """
            cv_results = pd.DataFrame(grid_search.cv_results_)
            rbf_results = cv_results[cv_results["param_svm__kernel"] == "rbf"]
            pivot = rbf_results.pivot_table(
                values="mean_test_score", index="param_svm__C", columns="param_svm__gamma"
            )
            plt.imshow(pivot, cmap="viridis", aspect="auto")
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xlabel("gamma")
            plt.ylabel("C")
            plt.title("Heatmap F1 CV — Kernel RBF")
            plt.colorbar(label="F1")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: perbandingan kernel

            Semua kernel berada dalam Pipeline yang sama agar scaling dipelajari hanya dari train set [ML-MODUL-01, hlm. 59].
            """,
        ),
        (
            "code",
            """
            kernel_cv_scores = {}
            kernel_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for kernel in ["linear", "poly", "rbf"]:
                candidate = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel=kernel, C=1)),
                ])
                kernel_cv_scores[kernel] = cross_val_score(
                    candidate, X_train, y_train, cv=kernel_cv, scoring="accuracy"
                ).mean()
            print("Rata-rata akurasi CV per kernel:", kernel_cv_scores)
            """,
        ),
        (
            "code",
            """
            selected_features = ["mean radius", "mean texture"]
            X_two = data.data[selected_features].to_numpy()
            y_two = data.target.to_numpy()
            scaler_two = StandardScaler()
            X_two_scaled = scaler_two.fit_transform(X_two)
            x_min, x_max = X_two_scaled[:, 0].min() - 0.5, X_two_scaled[:, 0].max() + 0.5
            y_min, y_max = X_two_scaled[:, 1].min() - 0.5, X_two_scaled[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
            boundary_configs = [
                {"kernel": "linear", "C": 1},
                {"kernel": "rbf", "C": 1, "gamma": "scale"},
                {"kernel": "rbf", "C": 10, "gamma": 0.1},
            ]
            fig, boundary_axes = plt.subplots(1, len(boundary_configs), figsize=(16, 4.5))
            for axis, config in zip(boundary_axes, boundary_configs):
                boundary_model = SVC(**config).fit(X_two_scaled, y_two)
                zz = boundary_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                axis.contourf(xx, yy, zz, alpha=0.25, cmap="coolwarm")
                axis.scatter(X_two_scaled[:, 0], X_two_scaled[:, 1], c=y_two, s=12, cmap="coolwarm")
                axis.set_xlabel(selected_features[0] + " (scaled)")
                axis.set_ylabel(selected_features[1] + " (scaled)")
                axis.set_title(str(config))
            fig.suptitle("Decision Boundary SVM — Perbandingan Kernel/C/Gamma")
            plt.tight_layout()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            C mengatur toleransi pelanggaran margin, gamma mengatur jangkauan pengaruh sampel pada RBF, dan kernel menentukan bentuk batas keputusan. Keduanya harus dituning melalui CV pada training set [ML-MODUL-01, hlm. 59].
            """,
        ),
    ],
)


register(
    9,
    "bab-09-ensemble-learning",
    "implementasi-bab-09-ensemble.ipynb",
    "Ensemble Learning",
    "63–64",
    [
        (
            "markdown",
            """
            # Implementasi Bab 9 — Ensemble Learning

            Decision Tree tunggal dibandingkan dengan Random Forest, AdaBoost, dan Gradient Boosting pada dataset Wine [ML-MODUL-01, hlm. 63–64].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_wine
            from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.tree import DecisionTreeClassifier

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            data = load_wine(as_frame=True)
            X_train, X_test, y_train, y_test = train_test_split(
                data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
            )
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=SEED),
                "Random Forest": RandomForestClassifier(
                    n_estimators=300, oob_score=True, random_state=SEED, n_jobs=-1
                ),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEED),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, random_state=SEED
                ),
            }
            for candidate in models.values():
                candidate.fit(X_train, y_train)
            """,
        ),
        (
            "code",
            """
            random_forest = models["Random Forest"]
            print("OOB Random Forest:", round(random_forest.oob_score_, 3))
            holdout_rows = []
            for name, candidate in models.items():
                score = accuracy_score(y_test, candidate.predict(X_test))
                holdout_rows.append((name, score))
            display(pd.DataFrame(holdout_rows, columns=["model", "test_accuracy"]).sort_values("test_accuracy", ascending=False))
            """,
        ),
        (
            "code",
            """
            importances = random_forest.feature_importances_
            indices = np.argsort(importances)[::-1][:8]
            plt.barh(
                [data.feature_names[index] for index in indices][::-1],
                importances[indices][::-1],
            )
            plt.xlabel("Feature Importance")
            plt.title("Feature Importance — Random Forest")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: pengaruh n_estimators

            OOB score menggunakan sampel yang tidak terambil pada bootstrap masing-masing pohon sebagai estimasi internal [ML-MODUL-01, hlm. 64].
            """,
        ),
        (
            "code",
            """
            estimator_rows = []
            for n_estimators in [10, 50, 100, 300, 500]:
                candidate = RandomForestClassifier(
                    n_estimators=n_estimators, oob_score=True, random_state=SEED, n_jobs=-1
                ).fit(X_train, y_train)
                estimator_rows.append((n_estimators, candidate.oob_score_))
            estimator_frame = pd.DataFrame(estimator_rows, columns=["n_estimators", "oob"])
            display(estimator_frame)
            selected_n_estimators = int(
                estimator_frame.sort_values("oob", ascending=False).iloc[0]["n_estimators"]
            )
            selected_forest = RandomForestClassifier(
                n_estimators=selected_n_estimators, random_state=SEED, n_jobs=-1
            ).fit(X_train, y_train)
            final_test_accuracy = accuracy_score(y_test, selected_forest.predict(X_test))
            print("n_estimators terpilih:", selected_n_estimators)
            print("Akurasi test final:", round(final_test_accuracy, 3))
            estimator_frame.plot(x="n_estimators", y="oob", marker="o")
            plt.ylim(0.7, 1.02)
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 2: Stratified k-Fold CV

            Empat model dibandingkan dengan split dan metrik `f1_macro` yang sama agar adil [ML-MODUL-01, hlm. 64].
            """,
        ),
        (
            "code",
            """
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_rows = []
            for name, candidate in models.items():
                scores = cross_val_score(candidate, data.data, data.target, cv=cv, scoring="f1_macro", n_jobs=-1)
                cv_rows.append((name, scores.mean(), scores.std()))
            display(pd.DataFrame(cv_rows, columns=["model", "mean_f1_macro", "std"]).sort_values("mean_f1_macro", ascending=False))
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Ensemble mengurangi kelemahan satu pohon melalui agregasi atau pembelajaran bertahap. Performa, stabilitas CV, biaya komputasi, dan interpretabilitas perlu dipertimbangkan bersama.
            """,
        ),
    ],
)


register(
    10,
    "bab-10-clustering",
    "implementasi-bab-10-clustering.ipynb",
    "Clustering",
    "69–70",
    [
        (
            "markdown",
            """
            # Implementasi Bab 10 — Clustering

            Segmentasi pelanggan sintetis dengan K-Means, Agglomerative Clustering, dan DBSCAN, disertai silhouette score dan pencarian k [ML-MODUL-01, hlm. 69–70].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from scipy.cluster.hierarchy import dendrogram, linkage
            from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
            from sklearn.metrics import silhouette_score
            from sklearn.preprocessing import StandardScaler

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            rng = np.random.default_rng(SEED)
            X = np.vstack([
                rng.normal([5, 200_000], [1.5, 30_000], (100, 2)),
                rng.normal([15, 800_000], [2, 50_000], (100, 2)),
                rng.normal([2, 50_000], [0.8, 15_000], (100, 2)),
            ])
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("Bentuk data:", X_scaled.shape)
            """,
        ),
        (
            "code",
            """
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=SEED)
            labels_kmeans = kmeans.fit_predict(X_scaled)
            agglomerative = AgglomerativeClustering(n_clusters=3, linkage="ward")
            labels_agglomerative = agglomerative.fit_predict(X_scaled)
            dbscan = DBSCAN(eps=0.3, min_samples=5)
            labels_dbscan = dbscan.fit_predict(X_scaled)

            kmeans_silhouette = silhouette_score(X_scaled, labels_kmeans)
            agglomerative_silhouette = silhouette_score(X_scaled, labels_agglomerative)
            dbscan_mask = labels_dbscan != -1
            dbscan_cluster_count = len(set(labels_dbscan[dbscan_mask]))
            dbscan_silhouette = (
                silhouette_score(X_scaled[dbscan_mask], labels_dbscan[dbscan_mask])
                if dbscan_cluster_count >= 2 else np.nan
            )
            cluster_summary = pd.DataFrame([
                ("K-Means", 3, kmeans_silhouette, 0),
                ("Hierarchical", 3, agglomerative_silhouette, 0),
                ("DBSCAN", dbscan_cluster_count, dbscan_silhouette, int((labels_dbscan == -1).sum())),
            ], columns=["model", "cluster_count", "silhouette", "noise_count"])
            selected_business_model = cluster_summary.sort_values(
                "silhouette", ascending=False, na_position="last"
            ).iloc[0]["model"]
            display(cluster_summary)
            print("Rekomendasi numerik awal:", selected_business_model)
            """,
        ),
        (
            "code",
            """
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for axis, labels, title in zip(
                axes,
                [labels_kmeans, labels_agglomerative, labels_dbscan],
                ["K-Means", "Hierarchical", "DBSCAN"],
            ):
                axis.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="tab10", s=20)
                axis.set_title(title)
                axis.set_xlabel("Frekuensi (scaled)")
                axis.set_ylabel("Nilai transaksi (scaled)")
            plt.tight_layout()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan 1: Elbow dan silhouette

            Inertia selalu turun saat k bertambah; silhouette membantu menilai kepadatan dan pemisahan cluster [ML-MODUL-01, hlm. 70].
            """,
        ),
        (
            "code",
            """
            k_values = range(2, 11)
            inertias = []
            silhouettes = []
            for k in k_values:
                candidate = KMeans(n_clusters=k, n_init=10, random_state=SEED)
                labels = candidate.fit_predict(X_scaled)
                inertias.append(candidate.inertia_)
                silhouettes.append(silhouette_score(X_scaled, labels))

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].plot(list(k_values), inertias, marker="o")
            axes[0].set(title="Elbow Method", xlabel="k", ylabel="Inertia")
            axes[1].plot(list(k_values), silhouettes, marker="o")
            axes[1].set(title="Silhouette", xlabel="k", ylabel="Score")
            plt.show()
            print("k silhouette terbaik:", list(k_values)[int(np.argmax(silhouettes))])
            """,
        ),
        (
            "code",
            """
            linked = linkage(X_scaled, method="ward")
            plt.figure(figsize=(12, 4))
            dendrogram(linked, truncate_mode="lastp", p=20)
            plt.title("Dendrogram Ringkas")
            plt.xlabel("Cluster gabungan")
            plt.ylabel("Jarak")
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Interpretasi bisnis

            `selected_business_model` memilih silhouette tertinggi sebagai rekomendasi numerik awal. K-Means cocok ketika cluster relatif bulat dan jumlahnya diketahui; hierarchical membantu membaca struktur bertingkat; DBSCAN mampu menandai pola padat dan noise. Keputusan bisnis akhir tetap harus mempertimbangkan kemudahan interpretasi segmen, jumlah noise, dan tujuan kampanye—bukan silhouette saja.
            """,
        ),
    ],
)


register(
    11,
    "bab-11-dimensionality-reduction",
    "implementasi-bab-11-dimensionality-reduction.ipynb",
    "Dimensionality Reduction",
    "74–75",
    [
        (
            "markdown",
            """
            # Implementasi Bab 11 — PCA, t-SNE, dan LDA

            Reduksi dimensi dataset Digits, visualisasi tiga metode, dan evaluasi kompresi PCA terhadap klasifikasi [ML-MODUL-01, hlm. 74–75].
            """,
        ),
        (
            "code",
            """
            import random
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from sklearn.datasets import load_digits
            from sklearn.decomposition import PCA
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.linear_model import LogisticRegression
            from sklearn.manifold import TSNE
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            """,
        ),
        (
            "code",
            """
            digits = load_digits()
            X, y = digits.data, digits.target
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("Dimensi asli:", X.shape)
            """,
        ),
        (
            "code",
            """
            pca = PCA(n_components=0.90, random_state=SEED)
            X_pca = pca.fit_transform(X_scaled)
            pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(X_scaled)
            X_tsne = TSNE(
                n_components=2, perplexity=30, random_state=SEED, init="pca", learning_rate="auto"
            ).fit_transform(X_scaled)
            lda = LinearDiscriminantAnalysis(n_components=2)
            X_lda = lda.fit_transform(X_scaled, y)
            print("Komponen untuk 90% variance:", pca.n_components_)
            print("Total variance:", round(pca.explained_variance_ratio_.sum(), 3))
            """,
        ),
        (
            "code",
            """
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
            for axis, embedding, title in zip(
                axes,
                [pca_2d, X_tsne, X_lda],
                ["PCA 2D", "t-SNE 2D", "LDA 2D"],
            ):
                scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap="tab10", s=10)
                axis.set_title(title)
            fig.colorbar(scatter, ax=axes, label="Digit")
            plt.show()
            """,
        ),
        (
            "code",
            """
            cumulative_variance = np.cumsum(PCA(random_state=SEED).fit(X_scaled).explained_variance_ratio_)
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
            plt.axhline(0.90, color="red", linestyle="--", label="90%")
            plt.xlabel("Jumlah komponen")
            plt.ylabel("Cumulative explained variance")
            plt.title("Scree/Cumulative Variance Plot")
            plt.legend()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan: dampak kompresi PCA

            Split dilakukan sebelum fitting scaler dan PCA agar test set tidak memengaruhi transformasi [ML-MODUL-01, hlm. 75].
            """,
        ),
        (
            "code",
            """
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=SEED, stratify=y
            )
            compression_cv_rows = []
            compression_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for n_components in [0.80, 0.90, 0.95, None]:
                compression_pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=SEED)),
                    ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
                ])
                cv_accuracy = cross_val_score(
                    compression_pipeline, X_train_raw, y_train, cv=compression_cv, scoring="accuracy"
                ).mean()
                compression_cv_rows.append((n_components, cv_accuracy))
            compression_frame = pd.DataFrame(
                compression_cv_rows, columns=["variance_target", "cv_accuracy"]
            )
            display(compression_frame)
            selected_components = compression_frame.sort_values("cv_accuracy", ascending=False).iloc[0]["variance_target"]
            if pd.isna(selected_components):
                selected_components = None
            best_compression_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=selected_components, random_state=SEED)),
                ("model", LogisticRegression(max_iter=3000, random_state=SEED)),
            ]).fit(X_train_raw, y_train)
            final_test_accuracy = accuracy_score(y_test, best_compression_pipeline.predict(X_test_raw))
            selected_dimensions = best_compression_pipeline.named_steps["pca"].n_components_
            print("Konfigurasi PCA terpilih:", selected_components)
            print("Jumlah dimensi:", selected_dimensions)
            print("Akurasi test final:", round(final_test_accuracy, 3))
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            PCA bersifat linear dan dapat mentransformasi data baru; t-SNE unggul untuk eksplorasi visual lokal tetapi bukan pipeline transformasi data baru; LDA memakai label untuk mencari proyeksi diskriminatif [ML-MODUL-01, hlm. 75].
            """,
        ),
    ],
)


register(
    13,
    "bab-13-cnn-transfer-learning",
    "implementasi-bab-13-cnn-transfer-learning.ipynb",
    "CNN dan Transfer Learning",
    "86–88",
    [
        (
            "markdown",
            """
            # Implementasi Bab 13 — CNN dan Transfer Learning

            Transfer learning MobileNetV2 dan CNN dari nol [ML-MODUL-01, hlm. 86–88]. Adaptasi utama: contoh folder citra diganti CIFAR-10 bawaan Keras agar notebook mandiri. Runtime pertama dapat mengunduh dataset dan bobot ImageNet.
            """,
        ),
        (
            "code",
            """
            import random
            import time
            import matplotlib.pyplot as plt
            import numpy as np
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            tf.keras.utils.set_random_seed(SEED)
            print("TensorFlow:", tf.__version__)
            """,
        ),
        (
            "code",
            """
            (X_train_all, y_train_all), (X_test_all, y_test_all) = keras.datasets.cifar10.load_data()
            y_train_all = y_train_all.ravel()
            y_test_all = y_test_all.ravel()

            rng = np.random.default_rng(SEED)
            train_indices = rng.choice(len(X_train_all), size=5000, replace=False)
            test_indices = rng.choice(len(X_test_all), size=1000, replace=False)
            X_train, y_train = X_train_all[train_indices], y_train_all[train_indices]
            X_test, y_test = X_test_all[test_indices], y_test_all[test_indices]
            print("Subset train/test:", X_train.shape, X_test.shape)
            """,
        ),
        (
            "code",
            """
            class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            for index, axis in enumerate(axes.ravel()):
                axis.imshow(X_train[index])
                axis.set_title(class_names[y_train[index]])
                axis.axis("off")
            plt.tight_layout()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Transfer learning MobileNetV2

            Citra CIFAR-10 berukuran 32×32 di-resize menjadi 96×96. `preprocess_input` mengikuti normalisasi MobileNetV2; base model dibekukan untuk feature extraction [ML-MODUL-01, hlm. 86–87].
            """,
        ),
        (
            "code",
            """
            image_size = (96, 96)
            base_model = MobileNetV2(
                input_shape=image_size + (3,), include_top=False, weights="imagenet"
            )
            base_model.trainable = False

            inputs = keras.Input(shape=(32, 32, 3))
            x = keras.layers.Resizing(*image_size)(inputs)
            x = preprocess_input(x)
            x = base_model(x, training=False)
            x = keras.layers.GlobalAveragePooling2D()(x)
            x = keras.layers.Dense(128, activation="relu")(x)
            x = keras.layers.Dropout(0.3)(x)
            outputs = keras.layers.Dense(10, activation="softmax")(x)
            transfer_model = keras.Model(inputs, outputs, name="mobilenetv2_cifar10")
            transfer_model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            transfer_model.summary()
            """,
        ),
        (
            "code",
            """
            early_stop = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
            start = time.perf_counter()
            transfer_history = transfer_model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=5,
                batch_size=64,
                callbacks=[early_stop],
                verbose=1,
            )
            transfer_seconds = time.perf_counter() - start
            transfer_loss, transfer_accuracy = transfer_model.evaluate(X_test, y_test, verbose=0)
            print(f"Transfer learning accuracy={transfer_accuracy:.3f}, waktu={transfer_seconds:.1f}s")
            """,
        ),
        (
            "code",
            """
            plt.plot(transfer_history.history["accuracy"], label="Training Accuracy")
            plt.plot(transfer_history.history["val_accuracy"], label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Akurasi")
            plt.title("MobileNetV2 Feature Extraction")
            plt.legend()
            plt.show()
            """,
        ),
        (
            "markdown",
            """
            ## Latihan: CNN sederhana dari nol

            Arsitektur mengikuti latihan modul dan memakai subset yang sama agar perbandingan waktu serta akurasi lebih adil [ML-MODUL-01, hlm. 87–88].
            """,
        ),
        (
            "code",
            """
            tf.keras.utils.set_random_seed(SEED)
            cnn_model = keras.Sequential([
                keras.layers.Input(shape=(32, 32, 3)),
                keras.layers.Rescaling(1.0 / 255),
                keras.layers.Conv2D(32, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(10, activation="softmax"),
            ])
            cnn_model.compile(
                optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            start = time.perf_counter()
            cnn_history = cnn_model.fit(
                X_train, y_train, validation_split=0.2, epochs=5, batch_size=64, verbose=1
            )
            cnn_seconds = time.perf_counter() - start
            cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
            print(f"CNN scratch accuracy={cnn_accuracy:.3f}, waktu={cnn_seconds:.1f}s")
            """,
        ),
        (
            "code",
            """
            comparison = {
                "Transfer Learning": {
                    "test_accuracy": transfer_accuracy,
                    "training_seconds": transfer_seconds,
                    "trainable_parameters": int(np.sum([np.prod(weight.shape) for weight in transfer_model.trainable_weights])),
                },
                "CNN Scratch": {
                    "test_accuracy": cnn_accuracy,
                    "training_seconds": cnn_seconds,
                    "trainable_parameters": int(np.sum([np.prod(weight.shape) for weight in cnn_model.trainable_weights])),
                },
            }
            comparison
            """,
        ),
        (
            "markdown",
            """
            ## Kesimpulan

            Transfer learning memanfaatkan fitur ImageNet dan hanya melatih classifier baru saat base dibekukan. CNN scratch lebih bebas menyesuaikan domain tetapi biasanya memerlukan lebih banyak data/epoch. Hasil subset kecil adalah demonstrasi akademik, bukan benchmark final.
            """,
        ),
    ],
)


if __name__ == "__main__":
    for spec in NOTEBOOK_SPECS:
        build_notebook(*spec)
