from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np      
import pandas as pd

def train_svm_from_csv(
    csv_path: str,
    model_path: str = "svm_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Načte CSV s daty, natrénuje SVM klasifikátor a uloží ho na disk.

    Předpokládaný formát CSV:
        filename,label,feat1,feat2,feat3,...

    - 'filename' bude ignorován
    - 'label' bude cílová proměnná (string -> interně kategorie)
    - ostatní sloupce budou použity jako numerické feature

    Parametry:
        csv_path   : cesta k CSV souboru
        model_path : kam uložit natrénovaný model (.joblib)
        test_size  : podíl dat pro test (např. 0.2 = 20 %)
        random_state: seed pro opakovatelnost
    """

    # 1) načtení dat
    print(f"Načítám data z: {csv_path}")
    df = pd.read_csv(csv_path)

    # zkontrolujeme, že tam jsou sloupce 'label'
    if "label" not in df.columns:
        raise ValueError("CSV neobsahuje sloupec 'label'.")

    # 2) oddělení X (features) a y (label)
    # odstraníme ne-numerické sloupce jako 'filename'
    non_feature_cols = ["filename", "label"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    # jen numerické sloupce:
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values  # stringy (neutral, smile, ...)

    print(f"Počet vzorků: {len(y)}")
    print(f"Počet feature: {X.shape[1]}")

    # 3) train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Trénovacích vzorků: {len(y_train)}")
    print(f"Testovacích vzorků: {len(y_test)}")

    # 4) pipeline: standardizace + SVM
    # StandardScaler normalizuje feature (důležité pro SVM)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10, gamma="scale")),
        ]
    )

    # 5) trénování
    print("Trénuji SVM...")
    clf.fit(X_train, y_train)

    # 6) vyhodnocení
    print("Vyhodnocuji na testovací sadě...")
    y_pred = clf.predict(X_test)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 7) uložení modelu
    joblib.dump(clf, model_path)
    print(f"Model uložen do: {model_path}")

    return clf


#import joblib
#import numpy as np

#clf = joblib.load("svm_face_expression.joblib")

# feature_vector = np.array([...])   # stejná dimenze jako při tréninku
#pred_label = clf.predict(feature_vector.reshape(1, -1))[0]
#print("Predikovaný výraz:", pred_label)
