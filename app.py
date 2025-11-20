"""Aplicatie web Flask pentru predicții de meciuri"""
from flask import Flask, render_template, request
import pandas as pd
from predict import MatchPredictor

app = Flask(__name__)

# Încarcă predictorul o singură dată la pornirea aplicației
predictor = MatchPredictor()

# Încarcă meta-datele despre echipe și sezoane
def load_metadata():
    stats = pd.read_csv('stats.csv')
    results = pd.read_csv('results.csv')
    current_season_teams = pd.read_csv('expected_points_2017_2018.csv')
    teams = sorted(current_season_teams['team'].dropna().unique())
    seasons = sorted(results['season'].dropna().astype(str).unique())
    return teams, seasons

# Încarcă expected points
def load_expected_points():
    try:
        expected_points = pd.read_csv('expected_points_2017_2018.csv')
        return expected_points
    except FileNotFoundError:
        print("Fișierul expected_points_2017_2018.csv nu a fost găsit.")
        return pd.DataFrame()  # Returnează un DataFrame gol dacă fișierul nu există
    except Exception as e:
        print(f"Eroare la încărcarea expected points: {e}")
        return pd.DataFrame()

TEAMS, SEASONS = load_metadata()
LAST_SEASON = SEASONS[-1] if SEASONS else None
EXPECTED_POINTS = load_expected_points()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    season_note = None
    error = None

    # Valori preselectate în formular
    selected_home = request.form.get('home_team', TEAMS[0] if TEAMS else '')
    selected_away = request.form.get('away_team', TEAMS[1] if len(TEAMS) > 1 else '')
    selected_season = request.form.get('season', LAST_SEASON)

    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        season = request.form.get('season') or LAST_SEASON

        if home_team == away_team:
            error = 'Echipele trebuie să fie diferite.'
        elif home_team not in TEAMS or away_team not in TEAMS:
            error = 'Alege echipe valide din listă.'
        else:
            season_for_stats = season
            if season not in SEASONS:
                season_for_stats = LAST_SEASON
                season_note = f"Sezonul {season} nu există în date, folosesc statisticile din {season_for_stats}."

            try:
                result = predictor.predict(home_team, away_team, season_for_stats)
                prediction = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'season_display': season,
                    'season_used': season_for_stats,
                    'season_note': season_note,
                    'result_text': result['prediction'],
                    'result_code': result['prediction_letter'],
                    'prob_home': result['probabilities']['Home'],
                    'prob_draw': result['probabilities']['Draw'],
                    'prob_away': result['probabilities']['Away']
                }
            except Exception as exc:
                error = str(exc)

    # Pregătește datele pentru tabelul de expected points
    expected_points_data = []
    has_expected_points = False
    
    if not EXPECTED_POINTS.empty:
        # Verifică structura fișierului CSV
        print("Coloanele disponibile în expected_points:", EXPECTED_POINTS.columns.tolist())
        
        # Încearcă să identifici coloanele corecte
        team_column = None
        points_column = None
        
        # Caută coloana pentru echipe
        for col in EXPECTED_POINTS.columns:
            if 'team' in col.lower() or 'nume' in col.lower() or 'echipa' in col.lower():
                team_column = col
            elif 'point' in col.lower() or 'punct' in col.lower() or 'expected' in col.lower():
                points_column = col
        
        # Dacă nu găsim coloanele prin nume, folosim primele două coloane
        if team_column is None and len(EXPECTED_POINTS.columns) >= 2:
            team_column = EXPECTED_POINTS.columns[0]
            points_column = EXPECTED_POINTS.columns[1]
        
        if team_column and points_column:
            try:
                # Curăță datele și sortează
                EXPECTED_POINTS[points_column] = pd.to_numeric(EXPECTED_POINTS[points_column], errors='coerce')
                sorted_expected_points = EXPECTED_POINTS.dropna(subset=[points_column]).sort_values(points_column, ascending=False)
                
                # Converteste DataFrame-ul în listă de dicționare pentru template
                expected_points_data = sorted_expected_points[[team_column, points_column]].to_dict('records')
                
                # Redenumește cheile pentru template
                for item in expected_points_data:
                    item['Team'] = item.pop(team_column)
                    item['ExpectedPoints'] = item.pop(points_column)
                
                has_expected_points = len(expected_points_data) > 0
                print(f"Au fost încărcate {len(expected_points_data)} echipe în tabel")
                
            except Exception as e:
                print(f"Eroare la procesarea expected points: {e}")
                has_expected_points = False
        else:
            print("Nu s-au putut identifica coloanele pentru echipe și puncte")
            has_expected_points = False

    return render_template(
        'index.html',
        teams=TEAMS,
        seasons=SEASONS,
        last_season=LAST_SEASON,
        prediction=prediction,
        error=error,
        selected_home=selected_home,
        selected_away=selected_away,
        selected_season=selected_season,
        expected_points_data=expected_points_data,
        has_expected_points=has_expected_points
    )


if __name__ == '__main__':
    app.run(debug=True)