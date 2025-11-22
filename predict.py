"""
Script pentru a face predictii pe meciuri noi
"""
import pandas as pd
import numpy as np
import pickle
import sys
import itertools

class MatchPredictor:
    def __init__(self, model_path='model_final.pkl', features_path='feature_columns.pkl', 
                 stats_path='databases/stats.csv'):
        """Incarca modelul si datele necesare"""
        print("Incarcare model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        # Mapare pentru rezultate (folosind aceeasi ca in training)
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        
        self.stats = pd.read_csv(stats_path)
        print(f"Model incarcat cu succes! {len(self.feature_columns)} features")
        print(f"Primele 10 features: {self.feature_columns[:10]}")
    
    def prev_season(self, season_str):
        """Din '2017-2018' -> '2016-2017'"""
        try:
            parts = season_str.split('-')
            start = int(parts[0])
            end = int(parts[1])
            return f"{start-1}-{end-1}"
        except Exception:
            try:
                year = int(season_str)
                return str(year-1)
            except Exception:
                return season_str

    def get_team_stats(self, team, season):
        """Obtine statisticile unei echipe pentru sezonul precedent"""
        prev_season = self.prev_season(season)
        team_stats = self.stats[
            (self.stats['team'] == team) & 
            (self.stats['season'] == prev_season)
        ]
        
        if team_stats.empty:
            # Fallback: cauta orice statistici pentru aceasta echipa
            team_all = self.stats[self.stats['team'] == team]
            if not team_all.empty:
                # Foloseste statisticile cele mai recente
                return team_all.iloc[-1]
            else:
                # Returneaza 0 pentru toate statisticile numerice
                numeric_cols = self.stats.select_dtypes(include=[np.number]).columns
                empty_stats = pd.Series(0, index=numeric_cols)
                empty_stats['season'] = prev_season
                return empty_stats
        
        return team_stats.iloc[0]

    def _get_numeric_stat_columns(self):
        """Returneaza coloanele numerice pentru features (exclude 'season', 'team')"""
        exclude_cols = ['season', 'team']
        numeric_cols = [col for col in self.stats.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        return numeric_cols

    def create_match_features(self, home_team, away_team, season):
        """Creeaza features pentru un meci folosind stats din sezonul precedent"""
        # Obtine stats pentru sezonul precedent
        home_stats = self.get_team_stats(home_team, season)
        away_stats = self.get_team_stats(away_team, season)
        
        numeric_cols = self._get_numeric_stat_columns()
        
        # Construieste features in acelasi format ca la antrenare
        features = {}
        
        # Home features
        for col in numeric_cols:
            home_val = home_stats[col] if pd.notna(home_stats.get(col, np.nan)) else 0.0
            features[f"{col}_home_prev"] = home_val
        
        # Away features  
        for col in numeric_cols:
            away_val = away_stats[col] if pd.notna(away_stats.get(col, np.nan)) else 0.0
            features[f"{col}_away_prev"] = away_val
        
        # Difference features
        for col in numeric_cols:
            home_val = features.get(f"{col}_home_prev", 0.0)
            away_val = features.get(f"{col}_away_prev", 0.0)
            features[f"{col}_diff_prev"] = home_val - away_val

        # Creeaza DataFrame
        features_df = pd.DataFrame([features])
        
        # Asigura-te ca toate coloanele sunt numerice
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        features_df = features_df.fillna(0.0)
        
        # Verifica daca avem toate coloanele necesare
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        extra_cols = set(features_df.columns) - set(self.feature_columns)
        
        if missing_cols:
            print(f"Avertisment: Lipsesc {len(missing_cols)} coloane. Se adauga cu 0...")
            for col in missing_cols:
                features_df[col] = 0.0
        
        if extra_cols:
            print(f"Avertisment: Elimin {len(extra_cols)} coloane extra...")
            features_df = features_df[self.feature_columns]
        
        # Reordoneaza coloanele pentru a corespunde cu antrenarea
        features_df = features_df[self.feature_columns]
        return features_df
    
    def predict(self, home_team, away_team, season='2017-2018'):
        """Face predictie pentru un meci"""
        # Creeaza features
        features = self.create_match_features(home_team, away_team, season)
        
        # Face predictie
        prediction_numeric = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Converteste predictia numerica inapoi in 'H', 'D', 'A'
        prediction_letter = self.reverse_mapping[prediction_numeric]
        
        # Mapare pentru afisare
        result_map = {'A': 'Away', 'D': 'Draw', 'H': 'Home'}
        
        return {
            'prediction': result_map[prediction_letter],
            'prediction_letter': prediction_letter,
            'probabilities': {
                'Home': float(probabilities[2]),  # H = 2
                'Draw': float(probabilities[1]),  # D = 1
                'Away': float(probabilities[0])   # A = 0
            }
        }
    
    def predict_batch(self, matches):
        """Face predictii pentru mai multe meciuri
        
        Args:matches: Lista de tupluri (home_team, away_team, season)
        """
        results = []
        for i, (home_team, away_team, season) in enumerate(matches):
            if i % 50 == 0:
                print(f"Procesat {i}/{len(matches)} meciuri...")
                
            pred = self.predict(home_team, away_team, season)
            results.append({
                'home_team': home_team,
                'away_team': away_team,
                'season': season,
                'prediction': pred['prediction'],
                'prediction_letter': pred['prediction_letter'],
                'prob_home': pred['probabilities']['Home'],
                'prob_draw': pred['probabilities']['Draw'],
                'prob_away': pred['probabilities']['Away']
            })
        
        return pd.DataFrame(results)


def save_all_matches_2017_2018_to_csv(output_file='output/all_matches_2017_2018.csv'):
    """Salveaza toate meciurile posibile pentru echipele din 2017-2018.csv"""
    predictor = MatchPredictor()
    
    # Incarca meciurile reale din sezonul 2017-2018 pentru a extrage echipele
    try:
        real_matches_2017_2018 = pd.read_csv('databases/2017-2018.csv')
        home_teams = real_matches_2017_2018['home_team'].unique().tolist()
        away_teams = real_matches_2017_2018['away_team'].unique().tolist()
        
        # Combina si elimina duplicate
        teams_2017_2018 = list(set(home_teams + away_teams))
        
        print(f"Gasite {len(teams_2017_2018)} echipe unice in 2017-2018.csv")
        print(f"Echipe: {sorted(teams_2017_2018)}")
        
    except FileNotFoundError:
        print("Eroare: Fisierul 2017-2018.csv nu a fost gasit!")
        print("Folosesc toate echipele din stats.csv...")
        teams_2017_2018 = predictor.stats['team'].unique().tolist()
    
    season = '2017-2018'
    
    print(f"Generare meciuri pentru {len(teams_2017_2018)} echipe din 2017-2018...")
    
    # Toate combinatiile posibile doar intre echipele din 2017-2018
    matches = [(home, away, season) for home, away in itertools.permutations(teams_2017_2018, 2)]
    
    print(f"Total meciuri de generat: {len(matches)}")
    
    # Foloseste predict_batch pentru eficienta
    df_results = predictor.predict_batch(matches)
    
    # Salveaza CSV
    df_results.to_csv(output_file, index=False)
    print(f"Fisier CSV creat: {output_file}")
    print(f"Meciuri generate: {len(df_results)}")
    
    # Afiseaza cateva statistici
    print(f"\nStatistici generate:")
    print(f"- Numar echipe: {len(teams_2017_2018)}")
    print(f"- Numar meciuri: {len(df_results)}")
    print(f"- Meciuri per echipa: {len(df_results) // len(teams_2017_2018)}")


def save_expected_points_2017_2018(csv_matches='output/all_matches_2017_2018.csv', output_file='output/expected_points_2017_2018.csv'):
    """Calculeaza punctele asteptate pentru fiecare echipa"""
    df = pd.read_csv(csv_matches)
    
    points_dict = {}
    
    for idx, row in df.iterrows():
        home_team = row['home_team']
        prob_home_win = row['prob_home']
        prob_draw = row['prob_draw']
        prob_away_win = row['prob_away']
        
        # Puncte asteptate pentru echipa gazda
        expected_points_home = prob_home_win * 3 + prob_draw * 1
        
        # Puncte asteptate pentru echipa oaspete
        expected_points_away = prob_away_win * 3 + prob_draw * 1
        
        # Adauga la total
        if home_team not in points_dict:
            points_dict[home_team] = 0.0
        points_dict[home_team] += expected_points_home
        
        away_team = row['away_team']
        if away_team not in points_dict:
            points_dict[away_team] = 0.0
        points_dict[away_team] += expected_points_away
    
    points_dict = {team: points for team, points in points_dict.items()}
    
    points_df = pd.DataFrame(list(points_dict.items()), columns=['team', 'expected_points'])
    points_df = points_df.sort_values('expected_points', ascending=False)
    
    points_df.to_csv(output_file, index=False)
    print(f"Tabela cu puncte asteptate a fost creata: {output_file}")
    print(f"\nTop 5 echipe:")
    print(points_df.head().to_string(index=False))

def main():
    """Exemplu de utilizare"""
    if len(sys.argv) < 2:
        print("Utilizare: python predict.py <home_team> <away_team>")
        print("\nExemplu: python predict.py 'Manchester United' 'Liverpool'")
        print("Sau pentru a genera toate meciurile pentru echipele din 2017-2018: python predict.py --all")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        # Genereaza toate meciurile pentru echipele din 2017-2018
        save_all_matches_2017_2018_to_csv()
        save_expected_points_2017_2018()
    else:
        # Predictie pentru un singur meci
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        season = '2017-2018'
        
        predictor = MatchPredictor()
        
        result = predictor.predict(home_team, away_team, season)
        
        print(f"\n{'='*60}")
        print(f"PREDICTIE MECI")
        print(f"{'='*60}")
        print(f"Acasa: {home_team}")
        print(f"Oaspete: {away_team}")
        print(f"Sezon: {season}")
        print(f"\nRezultat prezis: {result['prediction']} ({result['prediction_letter']})")
        print(f"\nProbabilitati:")
        print(f"  Home: {result['probabilities']['Home']:.1%}")
        print(f"  Draw: {result['probabilities']['Draw']:.1%}")
        print(f"  Away: {result['probabilities']['Away']:.1%}")

if __name__ == '__main__':
    main()