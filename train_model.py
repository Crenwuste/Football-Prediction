"""
Antrenarea unui model XGBoost pentru predictii de fotbal
- Foloseste pentru fiecare meci doar statistici din sezonul precedent.
- Training: toate sezoanele din results.csv (fara sezonul 2017-2018)
- Test: rezultate din 2017-2018
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pickle

def compute_season_weight(season, k=0.2):
    """Calculeaza weight-ul pentru un sezon folosind exponential decay"""
    try:
        start_year = int(season.split('-')[0])
        # Referinta: cel mai recent sezon din training
        most_recent_year = 2016  # pentru ca antrenam pe 2008-2016
        years_ago = most_recent_year - start_year
        return np.exp(-k * years_ago)
    except:
        return 1.0


class FootballPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.result_mapping = {'A': 0, 'D': 1, 'H': 2}
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        self.results = None
        self.stats = None
        self.test_results = None
    
    def add_experience_weights(self, features_df, stats_df, min_matches=10, max_penalty=0.5):
        """
        Reduce ponderea pentru echipele cu putine meciuri jucate (promovate recent)
        
        Args:
            features_df: DataFrame cu features pentru meciuri
            stats_df: DataFrame cu statistici pe echipe si sezoane
            min_matches: numarul minim de meciuri pentru a considera o echipa cu experienta
            max_penalty: penalitatea maxima (0.5 = jumatate din pondere)
        """
        print("Calcul ponderi bazate pe experienta echipelor...")
        
        # Calculeaza numarul total de meciuri jucate per echipa pana la fiecare sezon
        experience_data = {}
        
        # Pentru fiecare echipa si sezon, calculeaza meciurile jucate in sezoanele anterioare
        all_teams = stats_df['team'].unique()
        all_seasons = sorted(stats_df['season'].unique())
        
        for team in all_teams:
            team_data = stats_df[stats_df['team'] == team].sort_values('season')
            total_matches = 0
            for season in all_seasons:
                season_stats = team_data[team_data['season'] == season]
                if not season_stats.empty:
                    if 'matches_played' in season_stats.columns:
                        matches_played = season_stats.iloc[0]['matches_played']
                    elif all(col in season_stats.columns for col in ['wins', 'draws', 'losses']):
                        matches_played = (season_stats.iloc[0]['wins'] + 
                                        season_stats.iloc[0]['draws'] + 
                                        season_stats.iloc[0]['losses'])
                    else:
                        # Fallback: estimeaza 38 de meciuri per sezon pentru ligile europene
                        matches_played = 38
                    
                    total_matches += matches_played
                    experience_data[(team, season)] = total_matches
                else:
                    experience_data[(team, season)] = total_matches
        
        # Calculeaza ponderile pentru fiecare meci
        weights = []
        inexperienced_count = 0
        
        for idx, match in features_df.iterrows():
            season = match['season']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Obtine experienta ambelor echipe in sezonul precedent
            prev_season = self.prev_season(season)
            
            home_experience = experience_data.get((home_team, prev_season), 0)
            away_experience = experience_data.get((away_team, prev_season), 0)
            
            # Experienta minima dintre cele doua echipe
            min_experience = min(home_experience, away_experience)
            
            # Calculeaza penalitatea (1.0 = nicio penalitate, 0.5 = jumatate din pondere)
            if min_experience >= min_matches * 3:  # Multa experienta (3+ sezoane)
                penalty = 1.0
            elif min_experience >= min_matches:    # Moderata
                penalty = 0.8
            elif min_experience >= min_matches // 2:  # Putina experienta
                penalty = 0.6
            else:                                   # Foarte putina experienta
                penalty = max_penalty
                inexperienced_count += 1
            
            weights.append(penalty)
        
        print(f"Meciuri cu echipe neexperimentate: {inexperienced_count}/{len(weights)} ({(inexperienced_count/len(weights))*100:.1f}%)")
        return weights

    def add_experience_weights_simple(self, features_df, stats_df, min_matches=30):
        """
        Foloseste doar sezoanele jucate
        """
        print("Calcul ponderi bazate pe numarul de sezoane jucate...")
        
        # Calculeaza numarul de sezoane jucate per echipa pana la fiecare sezon
        seasons_played = {}
        
        all_teams = stats_df['team'].unique()
        all_seasons = sorted(stats_df['season'].unique())
        
        for team in all_teams:
            team_seasons = stats_df[stats_df['team'] == team]['season'].unique()
            cumulative_seasons = 0
            for season in all_seasons:
                if season in team_seasons:
                    cumulative_seasons += 1
                seasons_played[(team, season)] = cumulative_seasons
        
        weights = []
        inexperienced_count = 0
        
        for idx, match in features_df.iterrows():
            season = match['season']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Obtine numarul de sezoane jucate in sezonul precedent
            prev_season = self.prev_season(season)
            
            home_seasons = seasons_played.get((home_team, prev_season), 0)
            away_seasons = seasons_played.get((away_team, prev_season), 0)
            
            # Numarul minim de sezoane dintre cele doua echipe
            min_seasons = min(home_seasons, away_seasons)
            
            # Calculeaza penalitatea
            if min_seasons >= 3:      # Multa experienta (3+ sezoane)
                penalty = 1.0
            elif min_seasons == 2:    # 2 sezoane
                penalty = 0.8
            elif min_seasons == 1:    # 1 sezon
                penalty = 0.6
            else:                     # Promovata recent (0 sezoane)
                penalty = 0.4
                inexperienced_count += 1
            
            weights.append(penalty)
        
        print(f"Meciuri cu echipe promovate recent: {inexperienced_count}/{len(weights)} ({(inexperienced_count/len(weights))*100:.1f}%)")
        
        # Afiseaza distributia
        unique_weights, weight_counts = np.unique(weights, return_counts=True)
        print("Distributia ponderilor experienta:")
        for w, count in zip(unique_weights, weight_counts):
            print(f"  Weight {w}: {count} meciuri ({count/len(weights)*100:.1f}%)")
        
        return weights

    def add_head_to_head_weights(self, features_df, h2h_boost=2.0):
        """Adauga ponderi pentru meciurile head-to-head"""
        print("Adaugare ponderi head-to-head...")
        
        # Creeaza o copie pentru head-to-head analysis
        all_matches = features_df[['season', 'home_team', 'away_team']].copy()
        
        # Sorteaza dupa sezon pentru a gasi doar meciurile anterioare
        all_matches = all_matches.sort_values('season')
        
        weights = []
        h2h_count = 0
        
        for idx, match in features_df.iterrows():
            current_season = match['season']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Gaseste meciurile anterioare dintre aceste echipe (din sezoanele anterioare)
            previous_matches = all_matches[
                (all_matches['season'] < current_season) &
                (
                    ((all_matches['home_team'] == home_team) & (all_matches['away_team'] == away_team)) |
                    ((all_matches['home_team'] == away_team) & (all_matches['away_team'] == home_team))
                )
            ]
            
            # Aplica boost daca au jucat impotriva in trecut
            if len(previous_matches) > 0:
                weights.append(h2h_boost)
                h2h_count += 1
            else:
                weights.append(1.0)
        
        print(f"Ponderi head-to-head: {h2h_count} meciuri au boost H2H ({(h2h_count/len(weights))*100:.1f}%)")
        return weights
    
    def add_head_to_head_frequency_weights(self, features_df, max_boost=3.0):
        """Ponderi bazate pe frecventa meciurilor head-to-head"""
        print("Adaugare ponderi bazate pe frecventa head-to-head...")
        
        all_matches = features_df[['season', 'home_team', 'away_team']].copy()
        all_matches = all_matches.sort_values('season')
        
        weights = []
        weight_distribution = {1.0: 0, 1.5: 0, 2.0: 0, 2.5: 0}
        
        for idx, match in features_df.iterrows():
            current_season = match['season']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Numara meciurile anterioare
            previous_matches = all_matches[
                (all_matches['season'] < current_season) &
                (
                    ((all_matches['home_team'] == home_team) & (all_matches['away_team'] == away_team)) |
                    ((all_matches['home_team'] == away_team) & (all_matches['away_team'] == home_team))
                )
            ]
            
            h2h_count = len(previous_matches)
            
            # Aplica boost progresiv bazat pe numarul de meciuri anterioare
            if h2h_count == 0:
                weight = 1.0
            elif h2h_count <= 2:
                weight = 1.5  # boost moderat pentru putine meciuri
            elif h2h_count <= 5:
                weight = 2.0  # boost mediu
            else:
                weight = 2.5  # boost mare pentru rivalitati de lunga durata
                
            weight = min(weight, max_boost)
            weights.append(weight)
            weight_distribution[weight] = weight_distribution.get(weight, 0) + 1
        
        # Afiseaza distributia
        total_matches = len(weights)
        print("Distributia ponderilor H2H:")
        for weight, count in sorted(weight_distribution.items()):
            if count > 0:
                print(f"  Weight {weight}: {count} meciuri ({count/total_matches*100:.1f}%)")
        
        return weights
    
    # -------------------------
    # Loading
    # -------------------------
    def load_training_data(self, results_path='databases/results_train.csv', stats_path='databases/stats-max-2016.csv'):
        """Incarca datele de antrenare: results + stats."""
        print("Incarcare date de antrenare...")
        self.results = pd.read_csv(results_path)
        self.stats = pd.read_csv(stats_path)
        print(f"Loaded {len(self.results)} rezultate (train)")
        print(f"Loaded {len(self.stats)} statistici (per echipa / sezon)")
        print(f"Sezoane in results: {self.results['season'].unique()}")
        print(f"Sezoane in stats: {self.stats['season'].unique()}")
    
    def load_test_data(self, test_results_path='databases/results_2016-2017.csv'):
        """Incarca doar rezultatele din sezonul de test."""
        print("Incarcare date de test...")
        self.test_results = pd.read_csv(test_results_path)
        print(f"Loaded {len(self.test_results)} rezultate test")

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def prev_season(season_str):
        """
        Din '2017-2018' -> '2016-2017'. Daca formatul e diferit, incearca sa-l parseze.
        """
        try:
            parts = season_str.split('-')
            # suporta '2017-2018' -> scadem 1 din ambele parti
            start = int(parts[0])
            end = int(parts[1])
            return f"{start-1}-{end-1}"
        except Exception:
            # fallback: daca nu se poate, returneaza sezonul original minus 1 pe partea start (ex '2017'->'2016')
            try:
                year = int(season_str)
                return str(year-1)
            except Exception:
                return season_str

    def _get_numeric_stat_columns(self, stats_df):
        """Returneaza coloanele numerice utile pentru features."""
        exclude_cols = ['season', 'team']
        numeric_cols = [col for col in stats_df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        return numeric_cols

    # -------------------------
    # Feature building (general)
    # -------------------------
    def create_features_from(self, results_df, stats_df, save_path=None):
        """
        Creeaza features pentru un DataFrame de results folosind stats_df.
        IMPORTANT: stats_df trebuie sa contina intrari per team per season (dar NU pentru sezonul curent al results_df).
        Pentru fiecare meci din results_df folosim stats pentru sezonul precedent al meciului.
        """
        if results_df is None or stats_df is None:
            raise ValueError("results_df si stats_df trebuie sa fie furnizate.")

        print("\nCreare features (folosind stats din sezonul precedent)...")
        numeric_cols = self._get_numeric_stat_columns(stats_df)
        if not numeric_cols:
            raise ValueError("Nu am gasit coloane numerice in stats_df. Verifica fisierul stats.")

        rows = []
        for idx, match in results_df.iterrows():
            season = match.get('season')
            prev = self.prev_season(season) if pd.notna(season) else None

            home = match.get('home_team')
            away = match.get('away_team')

            # cautam stats pentru sezonul precedent
            home_stats = stats_df[(stats_df['team'] == home) & (stats_df['season'] == prev)]
            away_stats = stats_df[(stats_df['team'] == away) & (stats_df['season'] == prev)]

            # fallback la 0 daca nu exista stats
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

        # salvare CSV daca se cere
        if save_path:
            features_df.to_csv(save_path, index=False)
            print(f"Combined features saved to {save_path}")

        print(f"Features create: {features_df.shape[1] - (1 if 'result' in features_df.columns else 0)} features, {len(features_df)} meciuri")
        
        # Verifica variabilitatea
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
        """Pregateste X si y din features_df (fara split)."""
        if 'result' not in features_df.columns:
            raise ValueError("features_df trebuie sa contina coloana 'result' pentru a pregati y.")

        # COLOANE DE ELIMINAT - include toate coloanele de ponderi!
        columns_to_drop = [
            "result", "season", "home_team", "away_team", 
            "sample_weight", "season_weight"
        ]
        
        # Elimina doar coloanele care exista
        existing_columns_to_drop = [col for col in columns_to_drop if col in features_df.columns]
        X = features_df.drop(existing_columns_to_drop, axis=1)
        
        y = features_df["result"].map(self.result_mapping)

        # Asiguram ca toate coloanele sunt numerice
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
        """Antreneaza modelul XGBoost folosind RandomizedSearchCV pentru tuning automat."""
        print("\nAntrenare model XGBoost cu tuning automat...")

        # Spatiu de parametri pentru RandomizedSearch
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

        # Model de baza
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=42,
        )

        search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_grid,
            n_iter=100,
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
        print("Acuratetea CV:", search.best_score_)
        
        return search.best_params_


    def evaluate(self, X_test, y_test):
        """Evalueaza modelul pe un set de test."""
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat inca!")
        print("\nEvaluare model...")
        
        # Asiguram ca X_test are aceleasi coloane ca la antrenare
        missing_cols = set(self.feature_columns) - set(X_test.columns)
        extra_cols = set(X_test.columns) - set(self.feature_columns)
        
        if missing_cols:
            print(f"Avertisment: Lipsesc coloanele: {missing_cols}")
            for col in missing_cols:
                X_test[col] = 0.0
        
        if extra_cols:
            print(f"Avertisment: Elimin coloanele extra: {extra_cols}")
            X_test = X_test[self.feature_columns]
        
        # Reordoneaza coloanele pentru a corespunde cu antrenarea
        X_test = X_test[self.feature_columns]
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAcuratete: {acc:.4f}")
        print("\nRaport de clasificare:")
        print(classification_report(y_test, y_pred, target_names=['Away (A)', 'Draw (D)', 'Home (H)']))
        print("\nMatrice de confuzie:") 
        print(confusion_matrix(y_test, y_pred))
        
        # Verifica distributia predictiilor
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = {self.reverse_mapping[k]: v for k, v in zip(unique, counts)}
        print(f"\nDistributie predictii: {pred_distribution}")
        
        return acc

    def get_feature_importance(self, top_n=20):
        if self.model is None:
            raise ValueError("Modelul nu a fost antrenat inca!")

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
    # 1) Incarca datele pentru train
    # ----------------------------
    print("=== PAS 1: INCARCARE DATE ANTRENARE ===")
    predictor.load_training_data(results_path='databases/results_train.csv', stats_path='databases/stats-max-2016.csv')
    
    # ----------------------------
    # 2) Creeaza features pentru antrenare
    # ----------------------------
    print("\n=== PAS 2: CREARE FEATURES ANTRENARE ===")
    train_features = predictor.create_features_from(predictor.results, predictor.stats, 'databases/train_features.csv')
    
    # ----------------------------
    # 3) CALCUL PONDERI COMBINATE (SEZOANE + HEAD-TO-HEAD + EXPERIENTA)
    # ----------------------------
    print("\n=== PAS 3: CALCUL PONDERI COMBINATE ===")
    
    # Ponderi pentru sezoane recente
    train_features['season_weight'] = train_features['season'].apply(lambda s: compute_season_weight(s, 0.2))
    
    # Ponderi pentru head-to-head
    h2h_weights = predictor.add_head_to_head_weights(train_features, h2h_boost=2.0)
    
    # Ponderi pentru experienta echipelor
    experience_weights = predictor.add_experience_weights_simple(train_features, predictor.stats, min_matches=30)
    
    # COMBINA TOATE PONDERILE
    train_features['sample_weight'] = (
        train_features['season_weight'] * 
        h2h_weights * 
        experience_weights
    )
    
    # ----------------------------
    # 4) ANALIZA PONDERI FINALE
    # ----------------------------
    print(f"\n=== ANALIZA PONDERI FINALE ===")
    print(f"Min weight: {train_features['sample_weight'].min():.3f}")
    print(f"Max weight: {train_features['sample_weight'].max():.3f}")
    print(f"Mean weight: {train_features['sample_weight'].mean():.3f}")
    
    # Analiza detaliata
    weight_ranges = [
        (0, 0.5, "Foarte scazuta (promovate recent)"),
        (0.5, 0.8, "Scazuta (putina experienta)"),
        (0.8, 1.2, "Normala"),
        (1.2, 2.0, "Ridicata (H2H)"),
        (2.0, 10.0, "Foarte ridicata")
    ]
    
    print("\nDistributia ponderilor finale:")
    for min_w, max_w, desc in weight_ranges:
        count = len(train_features[
            (train_features['sample_weight'] >= min_w) & 
            (train_features['sample_weight'] < max_w)
        ])
        percentage = (count / len(train_features)) * 100
        print(f"  {desc}: {count} meciuri ({percentage:.1f}%)")
    
    # ----------------------------
    # 5) Pregateste X si y pentru antrenare
    # ----------------------------
    print("\n=== PAS 4: PREGATIRE DATE ANTRENARE ===")
    X_train, y_train = predictor.prepare_data(train_features)
    w_train = train_features['sample_weight']
    
    # Verifica distributia target
    unique_y, counts_y = np.unique(y_train, return_counts=True)
    y_distribution = {predictor.reverse_mapping[k]: v for k, v in zip(unique_y, counts_y)}
    print(f"Distributie target antrenare: {y_distribution}")
    
    # ----------------------------
    # 6) Incarca setul de validare
    # ----------------------------
    print("\n=== PAS 5: INCARCARE DATE VALIDARE ===")
    predictor.load_test_data(test_results_path='databases/results_2016-2017.csv')
    val_features = predictor.create_features_from(predictor.test_results, predictor.stats, 'databases/val_features.csv')
    
    X_val, y_val = predictor.prepare_data(val_features)
    
    # ----------------------------
    # 7) Antrenare cu tuning (CU PONDERI H2H)
    # ----------------------------
    print("\n=== PAS 6: ANTRENARE CU TUNING (CU PONDERI H2H) ===")
    best_params = predictor.train_with_tuning(X_train, y_train, sample_weight=w_train)
    
    # ----------------------------
    # 8) Evaluare pe validare (2016-2017)
    # ----------------------------
    print("\n=== PAS 7: EVALUARE PE VALIDARE (2016-2017) ===")
    val_accuracy = predictor.evaluate(X_val, y_val)
    
    # ----------------------------
    # 9) Re-antrenare pe TOATE datele (2008-2017) pentru testul final
    # ----------------------------
    print("\n=== PAS 8: RE-ANTRENARE PE TOATE DATELE ===")
    
    # Incarca toate datele (2008-2017)
    predictor.load_training_data(results_path='databases/results.csv', stats_path='databases/stats.csv')
    all_features = predictor.create_features_from(predictor.results, predictor.stats, 'databases/all_features.csv')
    
    # ADAUGA TOATE PONDERILE SI AICI
    all_features['season_weight'] = all_features['season'].apply(lambda s: compute_season_weight(s, 0.2))
    h2h_weights_all = predictor.add_head_to_head_weights(all_features, h2h_boost=2.0)
    experience_weights_all = predictor.add_experience_weights_simple(all_features, predictor.stats, min_matches=30)

    all_features['sample_weight'] = (
        all_features['season_weight'] * 
        h2h_weights_all * 
        experience_weights_all
    )
    
    # Folosim acelasi set de feature columns ca la antrenarea initiala
    X_all, y_all = predictor.prepare_data(all_features)
    w_all = all_features['sample_weight']
    
    # Antrenare finala cu cei mai buni parametri
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
    # 10) Test final pe datele SECRETE (2017-2018)
    # ----------------------------
    print("\n=== PAS 9: TEST FINAL PE DATE SECRETE (2017-2018) ===")
    predictor.load_test_data(test_results_path='databases/2017-2018.csv')
    test_features = predictor.create_features_from(predictor.test_results, predictor.stats, 'databases/test_features.csv')
    
    # NU avem sample_weight la test
    X_test, y_test = predictor.prepare_data(test_features)
    
    test_accuracy = predictor.evaluate(X_test, y_test)
    
    # ----------------------------
    # 11) Analiza modelului
    # ----------------------------
    print("\n=== PAS 10: ANALIZA MODELULUI ===")
    predictor.get_feature_importance()
    
    print(f"\n=== REZUMAT ===")
    print(f"Acuratete validare (2016-2017): {val_accuracy:.4f}")
    print(f"Acuratete test (2017-2018): {test_accuracy:.4f}")
    
    # Salvare model final
    with open('model_final.pkl', 'wb') as f:
        pickle.dump(predictor.model, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(predictor.feature_columns, f)
    print("Model final si feature columns salvate!")


if __name__ == '__main__':
    main()