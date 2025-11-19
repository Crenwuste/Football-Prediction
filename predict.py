"""
Script pentru a face predicții pe meciuri noi
"""
import pandas as pd
import numpy as np
import pickle
import sys

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
    
    def get_team_stats(self, team, season):
        """Obține statisticile unei echipe pentru un sezon"""
        team_stats = self.stats[
            (self.stats['team'] == team) & 
            (self.stats['season'] == season)
        ]
        
        if team_stats.empty:
            # Dacă nu găsește sezonul exact, încearcă să găsească ultimul sezon disponibil
            team_all_seasons = self.stats[self.stats['team'] == team]
            if not team_all_seasons.empty:
                # Folosește ultimul sezon disponibil
                team_stats = team_all_seasons.iloc[-1:]
                print(f"⚠ Avertisment: Nu s-a găsit sezonul {season} pentru {team}. "
                      f"Folosesc sezonul {team_stats['season'].values[0]}")
            else:
                raise ValueError(f"Nu s-au găsit statistici pentru echipa {team}")
        
        return team_stats.iloc[0]
    
    def create_match_features(self, home_team, away_team, season):
        """Creează features pentru un meci"""
        home_stats = self.get_team_stats(home_team, season)
        away_stats = self.get_team_stats(away_team, season)
        
        # Selectează coloanele numerice
        numeric_cols = self.stats.select_dtypes(include=[np.number]).columns.tolist()
        if 'season' in numeric_cols:
            numeric_cols.remove('season')
        
        features = {}
        for col in numeric_cols:
            home_val = home_stats[col] if pd.notna(home_stats[col]) else 0
            away_val = away_stats[col] if pd.notna(away_stats[col]) else 0
            features[f'{col}_diff'] = home_val - away_val
            features[f'{col}_home'] = home_val
            features[f'{col}_away'] = away_val
        
        # Creează DataFrame cu features
        features_df = pd.DataFrame([features])
        
        # Asigură-te că toate coloanele necesare există
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
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


def main():
    """Exemplu de utilizare"""
    if len(sys.argv) < 4:
        print("Utilizare: python predict.py <home_team> <away_team> <season>")
        print("\nExemplu: python predict.py 'Manchester United' 'Liverpool' '2008-2009'")
        print("\nSau folosește modul interactiv:")
        print("python predict.py")
        sys.exit(1)
    
    home_team = sys.argv[1]
    away_team = sys.argv[2]
    season = sys.argv[3]
    
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

