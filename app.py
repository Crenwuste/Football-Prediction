"""Aplicatie web Flask pentru predicții de meciuri"""
from flask import Flask, render_template, request
import pandas as pd
from predict import MatchPredictor
from train_model import WeightedXGBClassifier

app = Flask(__name__)

# Încarcă predictorul o singură dată la pornirea aplicației
predictor = MatchPredictor()

# Încarcă meta-datele despre echipe și sezoane
def load_metadata():
    stats = pd.read_csv('stats.csv')
    results = pd.read_csv('results.csv')
    teams = sorted(stats['team'].dropna().unique())
    seasons = sorted(results['season'].dropna().astype(str).unique())
    return teams, seasons

TEAMS, SEASONS = load_metadata()
LAST_SEASON = SEASONS[-1] if SEASONS else None

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

    return render_template(
        'index.html',
        teams=TEAMS,
        seasons=SEASONS,
        last_season=LAST_SEASON,
        prediction=prediction,
        error=error,
        selected_home=selected_home,
        selected_away=selected_away,
        selected_season=selected_season
    )


if __name__ == '__main__':
    app.run(debug=True)
