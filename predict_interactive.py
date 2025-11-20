"""
Script interactiv pentru predicții de meciuri
Permite introducerea unui meci și afișează șansele
"""
import sys
from predict import MatchPredictor
import pandas as pd

def get_available_teams():
    """Obține lista de echipe disponibile"""
    stats = pd.read_csv('stats.csv')
    return sorted(stats['team'].unique())

def get_available_seasons():
    """Obține lista de sezoane disponibile"""
    results = pd.read_csv('results.csv')
    return sorted(results['season'].unique())

def find_similar_team(team_name, available_teams):
    """Găsește echipe similare dacă numele nu se potrivește exact"""
    team_lower = team_name.lower()
    matches = [t for t in available_teams if team_lower in t.lower() or t.lower() in team_lower]
    return matches

def main():
    print("="*70)
    print("PREDICȚII MECIURI DE FOTBAL - MOD INTERACTIV")
    print("="*70)
    
    # Încarcă modelul
    print("\nÎncărcare model...")
    try:
        predictor = MatchPredictor()
    except FileNotFoundError as e:
        print(f"\n❌ Eroare: {e}")
        print("Asigură-te că ai antrenat modelul mai întâi cu: python train_model.py")
        sys.exit(1)
    
    # Obține echipele și sezoanele disponibile
    available_teams = get_available_teams()
    available_seasons = get_available_seasons()
    last_season = available_seasons[-1]
    
    print("Model încărcat cu succes!")
    print(f"📊 Model antrenat pe toate datele (sezoane: {available_seasons[0]} - {last_season})")
    print(f"📅 Ultimul sezon cu date: {last_season}\n")
    
    # Modul interactiv sau din argumente
    if len(sys.argv) >= 4:
        # Mod din linia de comandă
        home_team = sys.argv[1]
        away_team = sys.argv[2]
        season = sys.argv[3]
    else:
        # Mod interactiv
        print("Introdu datele meciului:")
        print(f"\nEchipe disponibile:")
        print(f"  {', '.join(available_teams[:10])}... (+ {len(available_teams)-10} altele)")
        
        print("\n" + "-"*70)
        home_team = input("\n📌 Echipa de acasă: ").strip()
        away_team = input("📌 Echipa oaspete: ").strip()
        
        print(f"\n💡 Poți introduce un sezon viitor (ex: 2018-2019, 2019-2020, etc.)")
        print(f"   Dacă sezonul nu există în date, se vor folosi statisticile din ultimul sezon disponibil ({last_season})")
        print(f"\nSezoane cu date disponibile: {', '.join(available_seasons[-5:])}")
        season_input = input(f"📅 Sezon pentru predicție (lăsă gol pentru {last_season}): ").strip()
        season = season_input if season_input else last_season
    
    # Verifică dacă echipele există
    if home_team not in available_teams:
        similar = find_similar_team(home_team, available_teams)
        if similar:
            print(f"\n⚠️  Echipă '{home_team}' nu a fost găsită exact.")
            print(f"   Echipe similare: {', '.join(similar[:5])}")
            if len(similar) == 1:
                home_team = similar[0]
                print(f"   Folosind: {home_team}")
            else:
                print("   Te rog să introduci numele exact al echipei.")
                sys.exit(1)
        else:
            print(f"\n❌ Echipă '{home_team}' nu a fost găsită.")
            print(f"   Echipe disponibile: {', '.join(available_teams[:10])}...")
            sys.exit(1)
    
    if away_team not in available_teams:
        similar = find_similar_team(away_team, available_teams)
        if similar:
            print(f"\n⚠️  Echipă '{away_team}' nu a fost găsită exact.")
            print(f"   Echipe similare: {', '.join(similar[:5])}")
            if len(similar) == 1:
                away_team = similar[0]
                print(f"   Folosind: {away_team}")
            else:
                print("   Te rog să introduci numele exact al echipei.")
                sys.exit(1)
        else:
            print(f"\n❌ Echipă '{away_team}' nu a fost găsită.")
            print(f"   Echipe disponibile: {', '.join(available_teams[:10])}...")
            sys.exit(1)
    
    # Verifică dacă sezonul există în date
    season_for_stats = season
    if season not in available_seasons:
        season_for_stats = last_season
        print(f"\n⚠️  Sezonul '{season}' nu există în date.")
        print(f"   Se vor folosi statisticile din ultimul sezon disponibil: {season_for_stats}")
        print(f"   (Modelul este antrenat pe toate datele, dar folosește statisticile recente pentru predicție)")
    
    # Face predicția
    print("\n" + "="*70)
    print("CALCULARE PREDICȚIE...")
    print("="*70)
    
    try:
        # Folosește sezonul pentru statistici (care poate fi diferit de sezonul introdus)
        result = predictor.predict(home_team, away_team, season_for_stats)
        
        # Afișează rezultatele
        print(f"\n🏟️  MECI: {home_team} vs {away_team}")
        print(f"📅 Sezon pentru predicție: {season}")
        if season != season_for_stats:
            print(f"📊 Statistici folosite din: {season_for_stats}")
        print("\n" + "-"*70)
        print("🎯 REZULTAT PREZIS:")
        print(f"   {result['prediction']} ({result['prediction_letter']})")
        print("\n" + "-"*70)
        print("📊 ȘANSE (PROBABILITĂȚI):")
        print(f"   🏠 {home_team:30} {result['probabilities']['Home']:6.2%}")
        print(f"   ⚖️  Egalitate                      {result['probabilities']['Draw']:6.2%}")
        print(f"   ✈️  {away_team:30} {result['probabilities']['Away']:6.2%}")
        print("-"*70)
        
        # Bară de progres vizuală pentru probabilități
        print("\n📈 Vizualizare șanse:")
        home_bar = "█" * int(result['probabilities']['Home'] * 50)
        draw_bar = "█" * int(result['probabilities']['Draw'] * 50)
        away_bar = "█" * int(result['probabilities']['Away'] * 50)
        
        print(f"   Home:   {home_bar} {result['probabilities']['Home']:.1%}")
        print(f"   Draw:   {draw_bar} {result['probabilities']['Draw']:.1%}")
        print(f"   Away:   {away_bar} {result['probabilities']['Away']:.1%}")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Eroare la generarea predicției: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

