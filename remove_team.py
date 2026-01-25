#!/usr/bin/env python3
"""
Script simple pour supprimer une équipe du leaderboard.
Usage: python remove_team.py <team_name>
"""
import json
import sys
from pathlib import Path
from datetime import datetime

def load_leaderboard():
    """Charge le leaderboard existant."""
    leaderboard_file = Path("leaderboard.json")
    if not leaderboard_file.exists():
        print("Erreur: leaderboard.json introuvable")
        sys.exit(1)
    
    with open(leaderboard_file, 'r') as f:
        return json.load(f)

def save_leaderboard(leaderboard):
    """Sauvegarde le leaderboard."""
    leaderboard["last_updated"] = datetime.now().isoformat()
    with open("leaderboard.json", 'w') as f:
        json.dump(leaderboard, f, indent=2)

def remove_team(team_name):
    """Supprime une équipe du leaderboard."""
    leaderboard = load_leaderboard()
    
    # Chercher et supprimer l'équipe
    original_count = len(leaderboard.get("submissions", []))
    original_teams = [entry.get("team") for entry in leaderboard.get("submissions", [])]
    
    leaderboard["submissions"] = [
        entry for entry in leaderboard.get("submissions", [])
        if entry.get("team") != team_name
    ]
    
    new_count = len(leaderboard["submissions"])
    
    if original_count == new_count:
        print(f"⚠️  Aucune équipe '{team_name}' trouvée dans le leaderboard")
        print(f"   Équipes disponibles: {', '.join(original_teams)}")
        return False
    else:
        # Sauvegarder
        save_leaderboard(leaderboard)
        print(f"✅ Équipe '{team_name}' supprimée du leaderboard")
        print(f"   {original_count} → {new_count} équipe(s)")
        
        # Régénérer le HTML
        try:
            from scripts.generate_leaderboard import generate_html
            generate_html(leaderboard)
            print("✅ leaderboard.html mis à jour")
        except Exception as e:
            print(f"⚠️  Impossible de régénérer le HTML: {e}")
        
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_team.py <team_name>")
        print("\nExemple: python remove_team.py team_sam_trad_ML_RandomForest")
        sys.exit(1)
    
    team_name = sys.argv[1]
    success = remove_team(team_name)
    sys.exit(0 if success else 1)

