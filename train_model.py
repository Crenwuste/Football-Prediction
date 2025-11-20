"""
Pipeline corect pentru antrenarea unui model XGBoost pentru predicții de fotbal
- Folosește pentru fiecare meci doar statistici din sezonul precedent.
- Training: toate sezoanele din results.csv (fără sezonul 2017-2018)
- Test: rezultate din 2017-2018 (folosind stats din 2016-2017)
"""
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pickle
import os


def compute_season_weight(season, k=0.2):
    """Calculează weight-ul pentru un sezon folosind exponential decay"""
    try:
        start_year = int(season.split('-')[0])
        # Referință: cel mai recent sezon din training
        most_recent_year = 2016  # pentru că antrenăm pe 2008-2016
        years_ago = most_recent_year - start_year
        return np.exp(-k * years_ago)
    except:
        return 1.0


class FootballPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        # Mapare pentru rezultate: 'A'=0, 'D'=1, 'H'=2
        self.result_mapping = {'A': 0, 'D': 1, 'H': 2}
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        # DataFrames loaded from disk
        self.results = None        # all training results (multiple seasons)
        self.stats = None          # per-team-per-season stats (all seasons)
        self.test_results = None   # results for last season (2017-2018)
    
    # -------------------------
    # Loading
    # -------------------------
    def load_training_data(self, results_path='databases/results_train.csv', stats_path='databases/stats-max-2016.csv'):
        """Încarcă datele de antrenare: results + stats."""
        print("Încărcare date de antrenare...")
        self.results = pd.read_csv(results_path)
        self.stats = pd.read_csv(stats_path)
        print(f"Loaded {len(self.results)} rezultate (train)")
        print(f"Loaded {len(self.stats)} statistici (per echipă / sezon)")
        print(f"Sezoane în results: {self.results['season'].unique()}")
        print(f"Sezoane în stats: {self.stats['season'].unique()}")
    
    def load_test_data(self, test_results_path='databases/results_2016-2017.csv'):
        """Încarcă doar rezultatele din sezonul de test."""
        print("Încărcare date de test...")
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
                return season_str

    def _get_numeric_stat_columns(self, stats_df):
        """Returnează coloanele numerice utile pentru features."""
        exclude_cols = ['season', 'team']
        numeric_cols = [col for col in stats_df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        return numeric_cols

    # -------------------------
    # Feature building (general)
    # -------------------------
    def create_features_from(self, results_df, stats_df, save_path=None):
        """
        Creează features pentru un DataFrame de results folosind stats_df.
        IMPORTANT: stats_df trebuie să conțină intrări per team per season (dar NU pentru sezonul curent al results_df).
        Pentru fiecare meci din results_df vom folosi stats pentru sezonul precedent al meciului.
        """
        if results_df is None or stats_df is None:
            raise ValueError("results_df și stats_df trebuie să fie furnizate.")

        print("\nCreare features (folosind stats din sezonul precedent)...")
        numeric_cols = self._get_numeric_stat_columns(stats_df)
        if not numeric_cols:
            raise ValueError("Nu am găsit coloane numerice în stats_df. Verifică fișierul stats.")

        rows = []
        for idx, match in results_df.iterrows():
            season = match.get('season')
            prev = self.prev_season(season) if pd.notna(season) else None

            home = match.get('home_team')
            away = match.get('away_team')

            # căutăm stats pentru sezonul precedent
            home_stats = stats_df[(stats_df['team'] == home) & (stats_df['season'] == prev)]
            away_stats = stats_df[(stats_df['team'] == away) & (stats_df['season'] == prev)]

            # fallback la 0 dacă nu există stats
            home_vals = home_stats.iloc[0][numeric_cols].to_dict() if not home_stats.empty else {c: 0.0 for c in numeric_cols}
            away_vals = away_stats.iloc[0][numeric_cols].to_dict() if not away_stats.empty else {c: 0.0 for c in numeric_cols}

            # construim features
            feat = {}
            for c in numeric_cols:
                h = home_vals.get(c, 0.0) if pd.notna(home_vals.get(c, np.nan)) else 0.0
                a = away_vals.get(c, 0.0) if pd.notna(away_vals.get(c, np.nan)) else 0.0
                feat[f"{c}_home_prev"] = h
                feat[f"{c}_away_prev"] = a
                feat[f"{c}_diff_prev"] = h - a

            # meta info
            feat['season'] = season
            feat['home_team'] = home
            feat['away_team'] = away
            if 'result' in results_df.columns:
                feat['result'] = match.get('result')

            rows.append(feat)

        features_df = pd.DataFrame(rows)

        if 'result' in features_df.columns:
            features_df = features_df.dropna(subset=['result'])

        features_df = features_df.fillna(0.0)

        # salvare CSV dacă se cere
        if save_path:
            features_df.to_csv(save_path, index=False)
            print(f"Combined features saved to {save_path}")

        print(f"Features create: {features_df.shape[1] - (1 if 'result' in features_df.columns else 0)} features, {len(features_df)} meciuri")
        
        # Verifică variabilitatea
        print("\nVerificare variabilitate features:")
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        for col in list(numeric_features)[:5]:  # Primele 5 pentru exemplu
            if col not in ['season']:
                unique_vals = features_df[col].nunique()
                print(f"  {col}: {unique_vals} valori unice")
        
        return features_df

    # -------------------------
    # Prepare (X, y)
    # -------------------------
    def prepare_data(self, features_df):
        """Pregătește X și y din features_df (fără split)."""
        if 'result' not in features_df.columns:
            raise ValueError("features_df trebuie să conțină coloana 'result' pentru a pregăti y.")

        # Coloane de eliminat - INCLUDE sample_weight aici pentru a-l elimina din features!
        columns_to_drop = ["result", "season", "home_team", "away_team", "sample_weight"]
        
        X = features_df.drop(columns_to_drop, axis=1, errors='ignore')
        y = features_df["result"].map(self.result_mapping)

        # Asiguram că toate coloanele sunt numerice
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0.0)

        self.feature_columns = X.columns.tolist()
        print(f"Feature columns: {len(self.feature_columns)}")
        return X, y

    # -------------------------
    # Training
    # -------------------------
    def train_with_tuning(self, X_train, y_train, sample_weight=None):
        """Antrenează modelul XGBoost folosind RandomizedSearchCV pentru tuning automat."""
        print("\nAntrenare model XGBoost cu tuning automat...")

        # Spațiu de parametri pentru RandomizedSearch
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.5],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 1.5, 2]
        }

        # Model de bază
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
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
        if sample_weight is not None:
            search.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            search.fit(X_train, y_train)

        self.model = search.best_estimator_
        print("Model antrenat cu succes!")
        print("Cei mai buni parametri:", search.best_params_)
        print("Acuratețea CV:", search.best_score_)
        
        return search.best_params_


    def evaluate(self, X_test, y_test):
        """Evaluează modelul pe un set de test."""
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")
        print("\nEvaluare model...")
        
        # Asigurăm că X_test are aceleași coloane ca la antrenare
        missing_cols = set(self.feature_columns) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(self.feature_columns)
        
        if missing_cols:
            print(f"Avertisment: Lipsesc coloanele: {missing_cols}")
            for col in missing_cols:
                X_test[col] = 0.0
        
        if extra_cols:
            print(f"Avertisment: Elimin coloanele extra: {extra_cols}")
            X_test = X_test[self.feature_columns]
        
        # Reordonează coloanele pentru a corespunde cu antrenarea
        X_test = X_test[self.feature_columns]
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAcuratețe: {acc:.4f}")
        print("\nRaport de clasificare:")
        print(classification_report(y_test, y_pred, target_names=['Away (A)', 'Draw (D)', 'Home (H)']))
        print("\nMatrice de confuzie:") 
        print(confusion_matrix(y_test, y_pred))
        
        # Verifică distribuția predicțiilor
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = {self.reverse_mapping[k]: v for k, v in zip(unique, counts)}
        print(f"\nDistribuție predicții: {pred_distribution}")
        
        return acc

    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat încă!")

        importance_dict = self.model.get_booster().get_score(importance_type='weight')
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)

        print(f"\nTop {top_n} features importante:")
        print(importance_df.head(top_n).to_string(index=False))
        return importance_df


# -------------------------
# main()
# -------------------------
def main():
    predictor = FootballPredictor()

    # ----------------------------
    # 1) Încarcă datele pentru train cu sezoanele 2008-2016.
    # ----------------------------
    print("=== PAS 1: ÎNCĂRCARE DATE ANTRENARE ===")
    predictor.load_training_data(results_path='databases/results_train.csv', stats_path='databases/stats-max-2016.csv')
    
    # ----------------------------
    # 2) Creează features pentru antrenare
    # ----------------------------
    print("\n=== PAS 2: CREARE FEATURES ANTRENARE ===")
    train_features = predictor.create_features_from(predictor.results, predictor.stats, 'databases/train_features.csv')
    
    # ----------------------------
    # 3) Adaugă ponderi pe sezoane
    # ----------------------------
    print("\n=== PAS 3: CALCUL PONDERI SEZOANE ===")
    train_features['sample_weight'] = train_features['season'].apply(lambda s: compute_season_weight(s, 0.2))
    print(f"Range ponderi: [{train_features['sample_weight'].min():.3f}, {train_features['sample_weight'].max():.3f}]")
    
    # ----------------------------
    # 4) Pregătește X și y pentru antrenare (EXCLUDE sample_weight din features)
    # ----------------------------
    print("\n=== PAS 4: PREGĂTIRE DATE ANTRENARE ===")
    X_train, y_train = predictor.prepare_data(train_features)
    w_train = train_features['sample_weight']  # păstrăm weights separat
    
    # Verifică distribuția target
    unique_y, counts_y = np.unique(y_train, return_counts=True)
    y_distribution = {predictor.reverse_mapping[k]: v for k, v in zip(unique_y, counts_y)}
    print(f"Distribuție target antrenare: {y_distribution}")
    
    # ----------------------------
    # 5) Încarcă setul de validare (sezonul 2016-2017) pentru tuning
    # ----------------------------
    print("\n=== PAS 5: ÎNCĂRCARE DATE VALIDARE ===")
    predictor.load_test_data(test_results_path='databases/results_2016-2017.csv')
    val_features = predictor.create_features_from(predictor.test_results, predictor.stats, 'databases/val_features.csv')
    
    # NU avem sample_weight la validare
    X_val, y_val = predictor.prepare_data(val_features)
    
    # ----------------------------
    # 6) Antrenare cu tuning pe 2008-2016, validare pe 2016-2017
    # ----------------------------
    print("\n=== PAS 6: ANTRENARE CU TUNING ===")
    best_params = predictor.train_with_tuning(X_train, y_train, sample_weight=w_train)
    
    # ----------------------------
    # 7) Evaluare pe validare (2016-2017)
    # ----------------------------
    print("\n=== PAS 7: EVALUARE PE VALIDARE (2016-2017) ===")
    val_accuracy = predictor.evaluate(X_val, y_val)
    
    # ----------------------------
    # 8) Re-antrenare pe TOATE datele (2008-2017) pentru testul final
    # ----------------------------
    print("\n=== PAS 8: RE-ANTRENARE PE TOATE DATELE ===")
    
    # Încarcă toate datele (2008-2017)
    predictor.load_training_data(results_path='databases/results.csv', stats_path='databases/stats.csv')
    all_features = predictor.create_features_from(predictor.results, predictor.stats, 'databases/all_features.csv')
    all_features['sample_weight'] = all_features['season'].apply(lambda s: compute_season_weight(s, 0.2))
    
    # Folosim același set de feature columns ca la antrenarea inițială
    X_all, y_all = predictor.prepare_data(all_features)
    w_all = all_features['sample_weight']  # păstrăm weights separat
    
    # Antrenare finală cu cei mai buni parametri
    final_model = xgb.XGBClassifier(
        **best_params,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=42,
    )
    
    final_model.fit(X_all, y_all, sample_weight=w_all)
    predictor.model = final_model
    
    # ----------------------------
    # 9) Test final pe datele SECRETE (2017-2018)
    # ----------------------------
    print("\n=== PAS 9: TEST FINAL PE DATE SECRETE (2017-2018) ===")
    predictor.load_test_data(test_results_path='databases/2017-2018.csv')
    test_features = predictor.create_features_from(predictor.test_results, predictor.stats, 'databases/test_features.csv')
    
    # NU avem sample_weight la test
    X_test, y_test = predictor.prepare_data(test_features)
    
    test_accuracy = predictor.evaluate(X_test, y_test)
    
    # ----------------------------
    # 10) Analiza modelului
    # ----------------------------
    print("\n=== PAS 10: ANALIZA MODELULUI ===")
    predictor.get_feature_importance()
    
    print(f"\n=== REZUMAT ===")
    print(f"Acuratețe validare (2016-2017): {val_accuracy:.4f}")
    print(f"Acuratețe test (2017-2018): {test_accuracy:.4f}")
    
    # Salvare model final
    with open('model_final.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(predictor.feature_columns, f)
    print("Model final și feature columns salvate!")


if __name__ == '__main__':
    main()