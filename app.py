from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TASKS = {
    "telephone": {
        "label": "Classification — téléphones",
        "file": DATA_DIR / "Telephone_data.csv",
        "target": "etat",
        "kind": "classification",
        "scaler": "minmax",
        "description": "Prédire la classe d'état du téléphone présente dans le dataset.",
    },
    "taxi": {
        "label": "Régression — trajets de taxi",
        "file": DATA_DIR / "taxi_trip_pricing.csv",
        "target": "Trip_Price",
        "kind": "regression",
        "scaler": "robust",
        "description": "Prédire le prix d'un trajet de taxi.",
    },
    "car_status": {
        "label": "Classification — voitures (Venant / Occasion)",
        "file": DATA_DIR / "Car_Data (1).csv",
        "target": "Venant_Occasion",
        "kind": "classification",
        "scaler": "robust",
        "description": "Prédire la classe Venant ou D'occasion.",
    },
    "car_price": {
        "label": "Régression — prix des voitures",
        "file": DATA_DIR / "Car_dataset.csv",
        "target": "Selling_Price",
        "kind": "regression",
        "scaler": "robust",
        "description": "Prédire Selling_Price à partir des caractéristiques de la voiture.",
    },
}


def model_specs(task_key):
    """Return the models and notebook-selected settings for one exercise."""
    if task_key == "telephone":
        return {
            "Régression logistique": (
                LogisticRegression(
                    C=0.46415888336127775,
                    penalty="l2",
                    max_iter=2000,
                ),
                {"C": 0.46415888336127775, "penalty": "l2"},
                None,
            ),
            "Arbre de décision": (
                DecisionTreeClassifier(
                    criterion="gini", max_depth=13, random_state=42
                ),
                {"criterion": "gini", "max_depth": 13},
                None,
            ),
            "Forêt aléatoire": (
                RandomForestClassifier(
                    criterion="gini",
                    max_depth=9,
                    n_estimators=30,
                    random_state=42,
                    n_jobs=-1,
                ),
                {"criterion": "gini", "max_depth": 9, "n_estimators": 30},
                None,
            ),
        }

    if task_key == "taxi":
        return {
            "Régression linéaire": (
                LinearRegression(),
                None,
                {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },
            ),
            "Ridge": (
                Ridge(),
                None,
                {
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                    "alpha": np.logspace(-8, 8, 100),
                },
            ),
            "Arbre de décision": (
                DecisionTreeRegressor(random_state=42),
                None,
                {
                    "max_depth": np.arange(1, 30),
                    "criterion": ["squared_error", "absolute_error"],
                },
            ),
            "Forêt aléatoire": (
                RandomForestRegressor(random_state=42),
                None,
                {
                    "n_estimators": np.arange(10, 100, 10),
                    "max_depth": np.arange(1, 15),
                },
            ),
        }

    if task_key == "car_status":
        return {
            "Régression logistique": (
                LogisticRegression(
                    C=0.00026366508987303583,
                    fit_intercept=True,
                    max_iter=2000,
                ),
                {"C": 0.00026366508987303583, "fit_intercept": True},
                None,
            ),
            "Ridge Classifier": (
                RidgeClassifier(alpha=10000.0, fit_intercept=True),
                {"alpha": 10000.0, "fit_intercept": True},
                None,
            ),
            "Arbre de décision": (
                DecisionTreeClassifier(
                    criterion="entropy", max_depth=6, random_state=42
                ),
                {"criterion": "entropy", "max_depth": 6},
                None,
            ),
            "Forêt aléatoire": (
                RandomForestClassifier(
                    n_estimators=90,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1,
                ),
                {"max_depth": 8, "n_estimators": 90},
                None,
            ),
        }

    if task_key == "car_price":
        return {
            "Régression linéaire": (
                LinearRegression(fit_intercept=True, positive=False),
                {"fit_intercept": True, "positive": False},
                None,
            ),
            "Ridge": (
                Ridge(
                    alpha=11.233240329780312,
                    fit_intercept=True,
                    positive=False,
                ),
                {
                    "alpha": 11.233240329780312,
                    "fit_intercept": True,
                    "positive": False,
                },
                None,
            ),
            "Arbre de décision": (
                DecisionTreeRegressor(
                    criterion="squared_error", max_depth=9, random_state=42
                ),
                {"criterion": "squared_error", "max_depth": 9},
                None,
            ),
            "Forêt aléatoire": (
                RandomForestRegressor(
                    n_estimators=90,
                    max_depth=13,
                    random_state=42,
                    n_jobs=-1,
                ),
                {"max_depth": 13, "n_estimators": 90},
                None,
            ),
        }

    raise ValueError(f"Travail inconnu : {task_key}")


@st.cache_resource(show_spinner="Entraînement des modèles de ce travail…")
def build_task_bundle(task_key):
    config = TASKS[task_key]
    data = pd.read_csv(config["file"])
    target = config["target"]
    if target not in data.columns:
        raise ValueError(f"La colonne cible « {target} » est absente du CSV.")

    # Une ligne sans cible ne peut pas servir à l'entraînement.
    target_missing = int(data[target].isna().sum())
    data = data.loc[data[target].notna()].copy()
    if config["kind"] == "classification":
        data[target] = data[target].astype(str).str.strip()

    x = data.drop(columns=[target]).copy()
    y = data[target].copy()
    if x.empty:
        raise ValueError("Aucune variable explicative n'a été trouvée.")

    categorical_features = list(
        x.select_dtypes(include=["object", "string", "category"]).columns
    )
    numeric_features = [col for col in x.columns if col not in categorical_features]
    category_modes = {}
    encoders = {}

    # Remplacer les catégories manquantes par le mode, comme dans la méthode du cours.
    for col in categorical_features:
        mode = x[col].dropna().astype(str).str.strip().mode()
        if mode.empty:
            raise ValueError(f"Impossible de déterminer le mode de « {col} ».")
        category_modes[col] = mode.iloc[0]
        values = x[col].astype("string").str.strip().fillna(category_modes[col])
        encoder = LabelEncoder()
        x[col] = encoder.fit_transform(values.astype(str))
        encoders[col] = encoder

    for col in numeric_features:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    # KNN ne sert que si les caractéristiques contiennent des valeurs manquantes.
    feature_imputer = None
    if x.isna().any().any():
        feature_imputer = KNNImputer(n_neighbors=5)
        x_values = feature_imputer.fit_transform(x)
        x = pd.DataFrame(x_values, columns=x.columns, index=x.index)

    feature_names = list(x.columns)
    scaler = MinMaxScaler() if config["scaler"] == "minmax" else RobustScaler()
    x_scaled = scaler.fit_transform(x[feature_names])

    specs = model_specs(task_key)
    models = {}
    selected_params = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, (estimator, fixed_params, param_grid) in specs.items():
        if param_grid is not None:
            grid = GridSearchCV(
                estimator,
                param_grid,
                cv=cv,
                n_jobs=-1,
                error_score="raise",
            )
            grid.fit(x_scaled, y)
            model = grid.best_estimator_
            selected_params[name] = grid.best_params_
        else:
            model = estimator.fit(x_scaled, y)
            selected_params[name] = fixed_params
        models[name] = model

    return {
        "task_key": task_key,
        "config": config,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "category_modes": category_modes,
        "encoders": encoders,
        "imputer": feature_imputer,
        "scaler": scaler,
        "models": models,
        "selected_params": selected_params,
        "training_rows": len(data),
        "target_missing_rows": target_missing,
        "training_data": data,
    }


def prepare_rows(input_df, bundle):
    missing_columns = [
        col for col in bundle["feature_names"] if col not in input_df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colonnes manquantes dans le CSV : " + ", ".join(missing_columns)
        )

    x = input_df[bundle["feature_names"]].copy()
    for col in bundle["categorical_features"]:
        values = x[col].astype("string").str.strip()
        values = values.replace("", pd.NA)
        values = values.fillna(bundle["category_modes"][col]).astype(str)
        encoder = bundle["encoders"][col]
        unknown = sorted(set(values) - set(encoder.classes_))
        if unknown:
            shown = ", ".join(unknown[:8]) + ("…" if len(unknown) > 8 else "")
            raise ValueError(
                f"Catégorie inconnue dans « {col} » : {shown}. "
                "Choisis une valeur présente dans le dataset d'entraînement."
            )
        x[col] = encoder.transform(values)

    for col in bundle["numeric_features"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    if x.isna().any().any():
        if bundle["imputer"] is None:
            cols = x.columns[x.isna().any()].tolist()
            raise ValueError(
                "Valeurs manquantes ou non numériques dans : " + ", ".join(cols)
            )
        values = bundle["imputer"].transform(x[bundle["feature_names"]])
        x = pd.DataFrame(values, columns=bundle["feature_names"])

    return bundle["scaler"].transform(x[bundle["feature_names"]])


def make_template(bundle):
    data = bundle["training_data"]
    row = {}
    for col in bundle["feature_names"]:
        if col in bundle["categorical_features"]:
            row[col] = bundle["category_modes"][col]
        else:
            median = pd.to_numeric(data[col], errors="coerce").median()
            if pd.api.types.is_integer_dtype(data[col]):
                row[col] = int(round(median))
            else:
                row[col] = float(median)
    return pd.DataFrame([row], columns=bundle["feature_names"])


def render_single(bundle, model, model_name):
    data = bundle["training_data"]
    defaults = make_template(bundle).iloc[0].to_dict()
    values = {}
    st.caption("Renseigne les caractéristiques d'un seul téléphone, trajet ou véhicule.")

    with st.form(f"simple_{bundle['task_key']}_{model_name}"):
        columns = st.columns(2)
        for i, col in enumerate(bundle["feature_names"]):
            with columns[i % 2]:
                if col in bundle["categorical_features"]:
                    options = sorted(
                        data[col].dropna().astype(str).str.strip().unique().tolist()
                    )
                    default = defaults[col]
                    if default not in options:
                        options.insert(0, default)
                    values[col] = st.selectbox(
                        col,
                        options,
                        index=options.index(default),
                        key=f"{bundle['task_key']}_{model_name}_{col}",
                    )
                else:
                    default = defaults[col]
                    if pd.api.types.is_integer_dtype(data[col]):
                        step = 1000 if col in {"prix", "Prix", "Kms_Driven"} else 1
                        values[col] = st.number_input(
                            col,
                            value=int(default),
                            step=step,
                            key=f"{bundle['task_key']}_{model_name}_{col}",
                        )
                    else:
                        values[col] = st.number_input(
                            col,
                            value=float(default),
                            step=0.1,
                            key=f"{bundle['task_key']}_{model_name}_{col}",
                        )

        submitted = st.form_submit_button("Lancer la prédiction")

    if submitted:
        row = pd.DataFrame([values], columns=bundle["feature_names"])
        try:
            x_scaled = prepare_rows(row, bundle)
            prediction = model.predict(x_scaled)[0]
        except ValueError as error:
            st.error(str(error))
            return

        if bundle["config"]["kind"] == "regression":
            st.success(f"Prix prédit : {prediction:,.2f}")
            st.caption("Valeur exprimée dans la même unité que la cible du dataset.")
        else:
            st.success(f"Classe prédite : **{prediction}**")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(x_scaled)[0]
                probability_df = pd.DataFrame(
                    {"Classe": model.classes_, "Probabilité": probabilities}
                ).sort_values("Probabilité", ascending=False)
                st.dataframe(
                    probability_df.style.format({"Probabilité": "{:.1%}"}),
                    use_container_width=True,
                    hide_index=True,
                )


def render_multiple(bundle, model, model_name):
    template = make_template(bundle)
    task_key = bundle["task_key"]
    st.download_button(
        "Télécharger un modèle de CSV",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"modele_{task_key}.csv",
        mime="text/csv",
        key=f"template_{task_key}_{model_name}",
    )
    st.caption(
        "Le CSV doit contenir toutes les colonnes de caractéristiques du modèle, "
        "avec une ligne par observation. La colonne cible est facultative."
    )
    uploaded = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"],
        key=f"upload_{task_key}_{model_name}",
    )
    if uploaded is None:
        return

    try:
        source = pd.read_csv(uploaded)
        x_scaled = prepare_rows(source, bundle)
        predictions = model.predict(x_scaled)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Impossible de produire les prédictions : {error}")
        return

    result = source.copy()
    if bundle["config"]["kind"] == "regression":
        result["Prix_predit"] = predictions
        target = bundle["config"]["target"]
        if target in source.columns:
            actual = pd.to_numeric(source[target], errors="coerce")
            result["Erreur_absolue"] = np.abs(actual - predictions)
    else:
        result["Classe_predite"] = predictions
        target = bundle["config"]["target"]
        if target in source.columns:
            actual = source[target].astype("string").str.strip()
            result["Classe_correcte"] = actual.eq(pd.Series(predictions, index=actual.index))

    st.success(f"Prédictions produites pour {len(result)} ligne(s).")
    st.dataframe(result.head(100), use_container_width=True)
    st.download_button(
        "Télécharger les résultats CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"predictions_{task_key}_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"results_{task_key}_{model_name}",
    )


def main():
    st.set_page_config(page_title="Modèles ML — quatre travaux", page_icon="🚘", layout="wide")
    st.title("Prédictions pour les quatre travaux de machine learning")
    st.write(
        "Choisis un travail et un modèle. Chaque modèle dispose d'une interface "
        "de prédiction simple et d'une interface de prédiction multiple par CSV."
    )

    task_key = st.sidebar.selectbox(
        "Travail / dataset",
        options=list(TASKS),
        format_func=lambda key: TASKS[key]["label"],
    )

    try:
        bundle = build_task_bundle(task_key)
    except Exception as error:
        st.error(f"Impossible de préparer ce travail : {error}")
        st.info("Vérifie que les quatre CSV se trouvent dans le dossier data.")
        st.stop()

    st.header(TASKS[task_key]["label"])
    st.write(TASKS[task_key]["description"])
    st.caption(f"Modèles entraînés sur {bundle['training_rows']:,} lignes disponibles.")
    if bundle["config"]["kind"] == "classification":
        observed_classes = sorted(bundle["training_data"][bundle["config"]["target"]].astype(str).unique())
        st.caption("Classes possibles : " + " · ".join(observed_classes))

    if task_key == "taxi":
        if bundle["target_missing_rows"]:
            st.info(
                f"{bundle['target_missing_rows']} ligne(s) sans Trip_Price ont été "
                "écartées de l'entraînement, car elles n'ont pas de cible. Les "
                "valeurs manquantes des variables explicatives sont imputées par "
                "KNN après remplissage des catégories par leur mode."
            )
    elif task_key == "car_status":
        st.warning(
            "Le CSV d'origine contient quelques prix très atypiques. Ils sont "
            "conservés tels quels ; vérifie-les à la source avant tout usage commercial."
        )

    model_name = st.sidebar.selectbox("Modèle", list(bundle["models"]))
    model = bundle["models"][model_name]
    with st.expander("Paramètres du modèle"):
        st.json(bundle["selected_params"][model_name])

    simple_tab, multiple_tab = st.tabs(
        ["Prédiction simple", "Prédiction multiple (CSV)"]
    )
    with simple_tab:
        render_single(bundle, model, model_name)
    with multiple_tab:
        render_multiple(bundle, model, model_name)

    with st.expander("À propos de la préparation"):
        st.write(
            "La préparation reprend les méthodes des notebooks : LabelEncoder "
            "pour les catégories, MinMaxScaler pour les téléphones et RobustScaler "
            "pour les trois autres jeux de données. Les valeurs extrêmes ne sont "
            "pas remplacées automatiquement."
        )
        st.write(
            "Les modèles du devoir ont été sélectionnés sur des découpages de "
            "validation/test. Pour l'application, ils sont ensuite réentraînés "
            "sur toutes les lignes disponibles."
        )


if __name__ == "__main__":
    main()
