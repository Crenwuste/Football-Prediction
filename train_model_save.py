"""
Pipeline corect pentru antrenarea unui model XGBoost pentru predicții de fotbal
- Folosește pentru fiecare meci doar statistici din sezonul precedent (fără leakage).
- Training: toate sezoanele din results.csv (fără sezonul 2017-2018)
- Test: rezultate din 2017-2018 (folosind stats din 2016-2017)
"""
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import xgboost as xgb
import pickle
import os


class WeightedXGBClassifier(xgb.XGBClassifier):
    def __init__(self, alpha=0.05, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        # dacă sample_weight nu e trecut, îl calculăm din coloana 'season'
        if 'sample_weight' not in kwargs:
            if 'season' in X.columns:
                kwargs['sample_weight'] = X['season'].apply(lambda s: compute_season_weight(s, self.alpha))
        # eliminăm coloana season înainte de fit
        return super().fit(X.drop(columns=['season'], errors='ignore'), y, **kwargs)


class FootballPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        # Mapare pentru rezultate: 'A'=0, 'D'=1, 'H'=2
        self.result_mapping = {'A': 0, 'D': 1, 'H': 2}
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        # DataFrames loaded from disk
        self.results = None        # all training results (multiple seasons, without last)
        self.stats = None          # per-team-per-season stats (all seasons except last)
        self.test_results = None   # results for last season (2017-2018)
    
    # -------------------------
    # Loading
    # -------------------------
    def load_training_data(self, results_path='results.csv', stats_path='stats.csv'):
        """Încarcă datele de antrenare: results + stats (stats NU conține sezonul de test)."""
        print("Încărcare date de antrenare...")
        self.results = pd.read_csv(results_path)
        self.stats = pd.read_csv(stats_path)
        print(f"Loaded {len(self.results)} rezultate (train)")
        print(f"Loaded {len(self.stats)} statistici (per echipă / sezon)")
    
    def load_test_data(self, test_results_path='2017-2018.csv'):
        """Încarcă doar rezultatele din sezonul de test (ex: 2017-2018)."""
        print("Încărcare date de test (ultimul sezon)...")
        self.test_results = pd.read_csv(test_results_path)
        print(f"Loaded {len(self.test_results)} rezultate test")

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def prev_season(season_str):
        """
        Din '2017-2018' -> '2016-2017'. Dacă formatul e diferit, încearcă să-l parseze.
        """
        try:
            parts = season_str.split('-')
            # suportă '2017-2018' -> scădem 1 din ambele părți
            start = int(parts[0])
            end = int(parts[1])
            return f"{start-1}-{end-1}"
        except Exception:
            # fallback: dacă nu se poate, returnează sezonul original minus 1 pe partea start (ex '2017'->'2016')
            try:
                year = int(season_str)
                return str(year-1)
            except Exception:
                return season_str  # last resort

    def _get_numeric_stat_columns(self, stats_df):
        """Returnează coloanele numerice utile pentru features (exclude 'season' dacă apare)."""
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'season' in numeric_cols:
            numeric_cols.remove('season')
        return numeric_cols

    # -------------------------
    # Feature building (general)
    # -------------------------
    def create_features_from(self, results_df, stats_df):
        """
        Creează features pentru un DataFrame de results folosind stats_df.
        IMPORTANT: stats_df trebuie să conțină intrări per team per season (dar NU pentru sezonul curent al results_df).
        Pentru fiecare meci din results_df vom folosi stats pentru sezonul precedent al meciului.
        """
        if results_df is None or stats_df is None:
            raise ValueError("results_df și stats_df trebuie să fie furnizate (nu None).")

        print("\nCreare features (folosind stats din sezonul precedent)...")
        numeric_cols = self._get_numeric_stat_columns(stats_df)
        if not numeric_cols:
            raise ValueError("Nu am găsit coloane numerice în stats_df. Verifică fișierul stats.")

        rows = []
        # iterăm rând cu rând (poate fi optimizat vectorial, dar claritate > micro-optimizare aici)
        for idx, match in results_df.iterrows():
            season = match.get('season')
            prev = self.prev_season(season) if pd.notna(season) else None

            home = match.get('home_team')
            away = match.get('away_team')

            # căutăm stats pentru sezonul precedent
            home_stats = stats_df[(stats_df['team'] == home) & (stats_df['season'] == prev)]
            away_stats = stats_df[(stats_df['team'] == away) & (stats_df['season'] == prev)]

            # dacă nu există stats pentru prev season, încercăm să folsim valorile medii pe sezonul precedent global (fallback)
            # dar în multe cazuri e ok să completăm cu 0 (sau mean)
            if home_stats.empty:
                home_vals = {c: 0.0 for c in numeric_cols}
            else:
                # luăm prima apariție (ar trebui să fie unică per echipă+season)
                home_vals = home_stats.iloc[0][numeric_cols].to_dict()

            if away_stats.empty:
                away_vals = {c: 0.0 for c in numeric_cols}
            else:
                away_vals = away_stats.iloc[0][numeric_cols].to_dict()

            # construim features pentru acest meci
            feat = {}
            for c in numeric_cols:
                h = home_vals.get(c, 0.0) if pd.notna(home_vals.get(c, np.nan)) else 0.0
                a = away_vals.get(c, 0.0) if pd.notna(away_vals.get(c, np.nan)) else 0.0
                feat[f"{c}_home_prev"] = h
                feat[f"{c}_away_prev"] = a
                feat[f"{c}_diff_prev"] = h - a

            # adăugăm coloanele meta (opțional)
            feat['season'] = season
            feat['home_team'] = home
            feat['away_team'] = away

            # target (poate lipsi la predicție, dar la training ar trebui să existe)
            if 'result' in results_df.columns:
                feat['result'] = match.get('result')

            rows.append(feat)

        features_df = pd.DataFrame(rows)

        # la training, eliminăm rândurile fără target
        if 'result' in features_df.columns:
            features_df = features_df.dropna(subset=['result'])

        # completăm NaN cu 0 (alege comportamentul preferat)
        features_df = features_df.fillna(0.0)

        print(f"Features create: {features_df.shape[1] - (1 if 'result' in features_df.columns else 0)} features, {len(features_df)} meciuri")
        return features_df

    # -------------------------
    # Convenience wrappers
    # -------------------------
    def create_training_features(self):
        """Creează features pentru datele de antrenare (folosind stats pentru sezonul precedent)."""
        return self.create_features_from(self.results, self.stats)

    def create_test_features(self):
        """Creează features pentru datele de test (test_results trebuie să fie încărcat)."""
        return self.create_features_from(self.test_results, self.stats)

    # -------------------------
    # Prepare (X, y)
    # -------------------------
    def prepare_data(self, features_df):
        """Pregătește X și y din features_df (fără split)."""
        if 'result' not in features_df.columns:
            raise ValueError("features_df trebuie să conțină coloana 'result' pentru a pregăti y (sau folosește doar pentru predict).")

        X = features_df.drop(["result", "season", "home_team", "away_team", "sample_weight"], axis=1, errors='ignore')
        y = features_df["result"].map(self.result_mapping)

        self.feature_columns = X.columns.tolist()
        return X, y

    # -------------------------
    # Training / Evaluate
    # -------------------------
    def train(self, X_train, y_train, early_stopping_rounds=10, use_eval_set=False, X_val=None, y_val=None):
        """Antrenează modelul XGBoost folosind RandomizedSearchCV pentru tuning automat."""
        print("\nAntrenare model XGBoost cu tuning automat...")

        # Spațiu de parametri pentru RandomizedSearch
        param_grid = {
            'alpha': [0.1, 0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.5, 0.6, 0.65, 0.7],
            'max_depth': [3, 4, 5, 6, 7],           # arbori nu prea adânci ca să nu overfit-uiască rapid
            'learning_rate': [0.01, 0.05, 0.1],      # valori clasice
            'n_estimators': [100, 200, 300, 400],    # fără early stopping, nu face 500+ estimatori
            'subsample': [0.6, 0.7, 0.8, 0.9],       # fracție de rânduri per arbore
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],# fracție de coloane per arbore
            'gamma': [0, 0.1, 0.5, 1],               # penalizare pentru split
            'min_child_weight': [1, 3, 5, 7],        # greutatea minimă pentru nod
            'reg_alpha': [0, 0.01, 0.1, 1],          # L1 regularization
            'reg_lambda': [0, 0.01, 0.1, 1],         # L2 regularization
            'max_delta_step': [0, 1, 3]              # pentru clase dezechilibrate
        }

        # Model de bază
        xgb_clf = WeightedXGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42
        )

        search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_grid,
            n_iter=50,
            scoring='accuracy',
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )

        # Folosim sample weights direct la fit
        search.fit(X_train, y_train)

        self.model = search.best_estimator_
        print("Model antrenat cu succes!")
        print("Cei mai buni parametri:", search.best_params_)
        print("Acuratețea CV:", search.best_score_)


    def evaluate(self, X_test, y_test):
        """Evaluează modelul pe un set de test."""
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        print("\nEvaluare model...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAcuratețe: {acc:.4f}")
        print("\nRaport de clasificare:")
        print(classification_report(y_test, y_pred, target_names=['Away (A)', 'Draw (D)', 'Home (H)']))
        print("\nMatrice de confuzie:") 
        print(confusion_matrix(y_test, y_pred))
        return acc

    # -------------------------
    # Predict single / batch
    # -------------------------
    def predict_match(self, home_team, away_team, season):
        """
        Prezice un meci necunoscut din 'season' folosind stats din sezonul precedent.
        Returnează mappingul rezultatelor (ex: 'H') și probabilitățile.
        """
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")

        # construim un mic features_df pentru acest meci
        single_row = pd.DataFrame([{
            'home_team': home_team,
            'away_team': away_team,
            'season': season
        }])

        feats = self.create_features_from(single_row, self.stats)
        # dacă nu are target, feats va avea doar coloanele features + meta
        X = feats.drop(["result", "season", "home_team", "away_team"], axis=1, errors='ignore')

        # asigurăm ordinea și coloanele folosite la antrenare
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_columns]

        probs = self.model.predict_proba(X)[0]
        pred_label = int(self.model.predict(X)[0])
        return {
            'prediction': self.reverse_mapping[pred_label],
            'probabilities': {
                'A': float(probs[0]),
                'D': float(probs[1]),
                'H': float(probs[2])
            }
        }

    # -------------------------
    # Save / Load model & metadata
    # -------------------------
    def save_model(self, model_path='football_model.pkl', features_path='feature_columns.pkl', mapping_path='result_mapping.pkl'):
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'result_mapping': self.result_mapping,
                'reverse_mapping': self.reverse_mapping
            }, f)
        print(f"\nModel salvat în {model_path}")
        print(f"Features salvate în {features_path}")
        print(f"Mapare salvată în {mapping_path}")

    def load_model(self, model_path='football_model.pkl', features_path='feature_columns.pkl', mapping_path='result_mapping.pkl'):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(features_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        with open(mapping_path, 'rb') as f:
            mm = pickle.load(f)
            self.result_mapping = mm['result_mapping']
            self.reverse_mapping = mm['reverse_mapping']
        print("Model și metadata încărcate.")

    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        print(f"\nTop {top_n} features importante:")
        print(feature_importance.head(top_n).to_string(index=False))
        return feature_importance


# -------------------------
# Exemplu de rulare: main()
# -------------------------
def main():
    predictor = FootballPredictor()

    # 1) Încarcă datele pentru train (sezoanele 1–8) și stats
    predictor.load_training_data(results_path='results_train.csv', stats_path='stats-max-2016.csv')
    train_features = predictor.create_training_features()

    # 2) Adaugă ponderi pe sezoane
    alpha_default = 0.3
    def compute_season_weight(season, alpha=alpha_default):
        years_ago = 2017 - int(season.split('-')[0])
        return max(0, 1.0 - alpha * years_ago)
    train_features['sample_weight'] = train_features['season'].apply(lambda s: compute_season_weight(s, alpha_default))

    # 3) Pregătește X și y
    X_train, y_train = predictor.prepare_data(train_features)

    # 4) Încarcă setul de validare (sezonul 9)
    predictor.load_test_data(test_results_path='results_2016-2017.csv')
    val_features = predictor.create_test_features()
    val_features['result'] = predictor.test_results['result']  # adaugă coloana rezultat
    X_val, y_val = predictor.prepare_data(val_features)


    # 5) Antrenare cu tuning (RandomizedSearchCV) pe primele 8 sezoane + validare pe 9
    predictor.train(
        X_train, y_train,
        use_eval_set=True,
        X_val=X_val,
        y_val=y_val
    )

    # 6) Re-antrenează modelul pe toate cele 9 sezoane (train + val)
    predictor.load_training_data(results_path='results.csv', stats_path='stats.csv')
    all_features = predictor.create_training_features()
    all_features['sample_weight'] = all_features['season'].apply(lambda s: compute_season_weight(s, alpha_default))
    X_all_train, y_all_train = predictor.prepare_data(all_features)
    all_weights = all_features['sample_weight']

    best_params = predictor.model.get_xgb_params()
    retrained_model = xgb.XGBClassifier(**best_params)
    retrained_model.fit(X_all_train, y_all_train, sample_weight=all_weights)
    predictor.model = retrained_model

    # 7) Evaluare pe TEST SECRET (sezonul 10)
    predictor.load_test_data(test_results_path='2017-2018.csv')
    test_features = predictor.create_test_features()
    X_test, y_test = predictor.prepare_data(test_features)
    predictor.evaluate(X_test, y_test)

    # 8) Importanța features
    predictor.get_feature_importance()

if __name__ == '__main__':
    main()
