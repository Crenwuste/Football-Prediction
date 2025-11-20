"""
Script pentru a face predicții pe meciuri noi
"""
from train_model import WeightedXGBClassifier
import pandas as pd
import numpy as np
import pickle
import sys
import itertools

class MatchPredictor:
    def __init__(self, model_path='football_model.pkl', features_path='feature_columns.pkl', 
                 mapping_path='result_mapping.pkl', stats_path='stats.csv'):
        """Încarcă modelul și datele necesare"""
        print("Încărcare model...")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(features_path, 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        # Încarcă maparea rezultatelor
        try:
            with open(mapping_path, 'rb') as f:
                mapping_data = pickle.load(f)
                self.reverse_mapping = mapping_data['reverse_mapping']
        except FileNotFoundError:
            # Fallback la maparea implicită dacă fișierul nu există
            self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        
        self.stats = pd.read_csv(stats_path)
        print("Model încărcat cu succes!")
    
    def get_team_stats(self, team, season=None):
        """Obține statisticile unei echipe pentru sezonul 2017-2018 sau le estimează pe baza mediei istorice."""
        season = '2017-2018'
        team_stats = self.stats[
            (self.stats['team'] == team) & 
            (self.stats['season'] == season)
        ]
        
        if team_stats.empty:
            # Creează statistici estimate folosind media numericelor
            team_all = self.stats[self.stats['team'] == team]
            avg_stats = team_all.select_dtypes(include=[np.number]).mean()
            avg_stats['season'] = season
            return avg_stats
        return team_stats.iloc[0]

    def create_match_features(self, home_team, away_team, season):
        """Creează features pentru un meci (optimizat fără fragmentare)"""
        home_stats = self.get_team_stats(home_team, season)
        away_stats = self.get_team_stats(away_team, season)
        
        # Selectează coloanele numerice
        numeric_cols = self.stats.select_dtypes(include=[np.number]).columns.tolist()
        if 'season' in numeric_cols:
            numeric_cols.remove('season')
        
        # Construiește dicționarul cu features
        features = {}
        for col in numeric_cols:
            home_val = home_stats[col] if pd.notna(home_stats[col]) else 0
            away_val = away_stats[col] if pd.notna(away_stats[col]) else 0
            features[f'{col}_diff'] = home_val - away_val
            features[f'{col}_home'] = home_val
            features[f'{col}_away'] = away_val
        
        # Creează DataFrame cu features
        features_df = pd.DataFrame([features])
        
        # Adaugă toate coloanele lipsă simultan pentru a evita fragmentarea
        missing_cols = [col for col in self.feature_columns if col not in features_df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=features_df.index, columns=missing_cols)
            features_df = pd.concat([features_df, missing_df], axis=1)
        
        # Reordonează coloanele în ordinea corectă
        features_df = features_df[self.feature_columns]
        
        return features_df
    
    def predict(self, home_team, away_team, season):
        """Face predicție pentru un meci"""
        # Creează features
        features = self.create_match_features(home_team, away_team, season)
        
        # Face predicție (returnează 0, 1, sau 2)
        prediction_numeric = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        # Convertește predicția numerică înapoi în 'H', 'D', 'A'
        prediction_letter = self.reverse_mapping[prediction_numeric]
        
        # Mapare pentru afișare
        result_map = {'A': 'Away', 'D': 'Draw', 'H': 'Home'}
        
        return {
            'prediction': result_map[prediction_letter],
            'prediction_letter': prediction_letter,
            'probabilities': {
                'Home': probabilities[2],  # H = 2
                'Draw': probabilities[1],  # D = 1
                'Away': probabilities[0]   # A = 0
            }
        }
    
    def predict_batch(self, matches):
        """Face predicții pentru mai multe meciuri
        
        Args:
            matches: Listă de tupluri (home_team, away_team, season)
        """
        results = []
        for home_team, away_team, season in matches:
            pred = self.predict(home_team, away_team, season)
            results.append({
                'home_team': home_team,
                'away_team': away_team,
                'season': season,
                'prediction': pred['prediction'],
                'prob_home': pred['probabilities']['Home'],
                'prob_draw': pred['probabilities']['Draw'],
                'prob_away': pred['probabilities']['Away']
            })
        
        return pd.DataFrame(results)


def save_all_matches_2017_2018_to_csv(output_file='all_matches_2017_2018.csv'):
    predictor = MatchPredictor()
    
    # Lista tuturor echipelor din stats
    teams = predictor.stats['team'].unique().tolist()
    season = '2017-2018'
    
    # Toate combinațiile posibile (home ≠ away)
    matches = [(home, away, season) for home, away in itertools.permutations(teams, 2)]
    
    # Listă pentru stocarea rezultatelor
    results = []
    for home, away, season in matches:
        pred = predictor.predict(home, away, season)
        results.append({
            'home_team': home,
            'away_team': away,
            'season': season,
            'prob_home': pred['probabilities']['Home'],
            'prob_draw': pred['probabilities']['Draw'],
            'prob_away': pred['probabilities']['Away']
        })
    
    # Creează DataFrame și salvează CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Fișier CSV creat: {output_file}")


def save_expected_points_2017_2018(csv_matches='all_matches_2017_2018.csv', output_file='expected_points_2017_2018.csv'):
    df = pd.read_csv(csv_matches)
    
    points_dict = {}
    for idx, row in df.iterrows():
        home_team = row['home_team']
        prob_home_win = row['prob_home']
        prob_draw = row['prob_draw']
        
        expected_points = prob_home_win * 3 + prob_draw * 1
        if home_team not in points_dict:
            points_dict[home_team] = 0
        points_dict[home_team] += expected_points
    
    # Rotunjire la int după ce s-au adunat toate punctele
    points_dict = {team: int(round(points)) for team, points in points_dict.items()}
    
    points_df = pd.DataFrame(list(points_dict.items()), columns=['team', 'expected_points'])
    points_df = points_df.sort_values('expected_points', ascending=False)
    
    points_df.to_csv(output_file, index=False)
    print(f"Tabela cu puncte așteptate a fost creată: {output_file}")

def main():
    """Exemplu de utilizare"""
    if len(sys.argv) < 3:
        print("Utilizare: python predict.py <home_team> <away_team> >")
        print("\nExemplu: python predict.py 'Manchester United' 'Liverpool' '2008-2009'")
        print("python predict.py")
        sys.exit(1)
    
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    season = season = '2017-2018'
    
    predictor = MatchPredictor()
    
    result = predictor.predict(home_team, away_team, season)
    
    print(f"\n{'='*60}")
    print(f"PREDICȚIE MECI")
    print(f"{'='*60}")
    print(f"Acasă: {home_team}")
    print(f"Oaspete: {away_team}")
    print(f"Sezon: {season}")
    print(f"\nRezultat prezis: {result['prediction']}")
    print(f"\nProbabilități:")
    print(f"  Home: {result['probabilities']['Home']:.2%}")
    print(f"  Draw: {result['probabilities']['Draw']:.2%}")
    print(f"  Away: {result['probabilities']['Away']:.2%}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
    save_all_matches_2017_2018_to_csv()
    save_expected_points_2017_2018()
