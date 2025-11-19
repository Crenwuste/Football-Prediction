"""
Script pentru antrenarea unui model XGBoost pentru predicții de fotbal
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os

class FootballPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        # Mapare pentru rezultate: 'A'=0, 'D'=1, 'H'=2
        self.result_mapping = {'A': 0, 'D': 1, 'H': 2}
        self.reverse_mapping = {0: 'A', 1: 'D', 2: 'H'}
        
    def load_data(self, results_path='results.csv', stats_path='stats.csv'):
        """Încarcă datele din CSV-uri"""
        print("Încărcare date...")
        self.results = pd.read_csv(results_path)
        self.stats = pd.read_csv(stats_path)
        print(f"Rezultate încărcate: {len(self.results)} meciuri")
        print(f"Statistici încărcate: {len(self.stats)} înregistrări")
        
    def create_features(self):
        """Creează features pentru fiecare meci combinând statisticile echipelor"""
        print("\nCreare features...")
        
        # Merge results cu stats pentru echipa de acasă
        df = self.results.merge(
            self.stats,
            left_on=['home_team', 'season'],
            right_on=['team', 'season'],
            how='left',
            suffixes=('', '_home')
        )
        
        # Merge cu stats pentru echipa oaspete
        df = df.merge(
            self.stats,
            left_on=['away_team', 'season'],
            right_on=['team', 'season'],
            how='left',
            suffixes=('_home', '_away')
        )
        
        # Elimină duplicatele de coloane
        df = df.drop(columns=['team_home', 'team_away', 'season_home', 'season_away'], errors='ignore')
        
        # Selectează coloanele numerice pentru features
        numeric_cols = self.stats.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('season') if 'season' in numeric_cols else None
        
        # Creează features diferențiale (home - away)
        feature_data = []
        for idx, row in df.iterrows():
            features = {}
            
            # Pentru fiecare metrică, creează diferența home - away
            for col in numeric_cols:
                home_col = f'{col}_home'
                away_col = f'{col}_away'
                
                if home_col in df.columns and away_col in df.columns:
                    home_val = row[home_col] if pd.notna(row[home_col]) else 0
                    away_val = row[away_col] if pd.notna(row[away_col]) else 0
                    features[f'{col}_diff'] = home_val - away_val
                    features[f'{col}_home'] = home_val
                    features[f'{col}_away'] = away_val
            
            feature_data.append(features)
        
        features_df = pd.DataFrame(feature_data)
        
        # Adaugă rezultatul ca target
        features_df['result'] = df['result'].values
        
        # Elimină rândurile cu valori lipsă
        features_df = features_df.dropna()
        
        print(f"Features create: {features_df.shape[1] - 1} features, {len(features_df)} meciuri")
        
        return features_df
    
    def prepare_data(self, features_df):
        """Pregătește datele pentru antrenare"""
        # Separă features și target
        X = features_df.drop('result', axis=1)
        y = features_df['result']
        
        # Convertește rezultatele din 'H', 'D', 'A' în 0, 1, 2
        y_encoded = y.map(self.result_mapping)
        
        # Salvează numele coloanelor
        self.feature_columns = X.columns.tolist()
        
        # Împarte în train și test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Antrenează modelul XGBoost"""
        print("\nAntrenare model XGBoost...")
        
        # Parametrii pentru XGBoost
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        # Antrenare cu validare
        if X_test is not None and y_test is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Model antrenat cu succes!")
        
    def evaluate(self, X_test, y_test):
        """Evaluează modelul"""
        print("\nEvaluare model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAcuratețe: {accuracy:.4f}")
        print("\nRaport de clasificare:")
        print(classification_report(y_test, y_pred, target_names=['Away (A)', 'Draw (D)', 'Home (H)']))
        print("\nMatrice de confuzie:")
        print("(Away, Draw, Home)")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_path='football_model.pkl', features_path='feature_columns.pkl', mapping_path='result_mapping.pkl'):
        """Salvează modelul, coloanele de features și maparea rezultatelor"""
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
    
    def get_feature_importance(self, top_n=20):
        """Afișează importanța features-urilor"""
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


def main():
    """Funcția principală"""
    predictor = FootballPredictor()
    
    # Încarcă datele
    predictor.load_data()
    
    # Creează features
    features_df = predictor.create_features()
    
    # Pregătește datele
    X_train, X_test, y_train, y_test = predictor.prepare_data(features_df)
    
    # Antrenează modelul
    predictor.train(X_train, y_train, X_test, y_test)
    
    # Evaluează modelul
    predictor.evaluate(X_test, y_test)
    
    # Afișează importanța features-urilor
    predictor.get_feature_importance()
    
    # Salvează modelul
    predictor.save_model()
    
    print("\n✓ Antrenare completă!")


if __name__ == '__main__':
    main()

