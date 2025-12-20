# 🚀 Guide de Configuration GitHub et Leaderboard

Ce guide vous explique comment mettre en place votre challenge sur GitHub avec un leaderboard automatique.

## 📋 Étapes de Configuration

### 1. Créer le Repository GitHub

1. **Créer un nouveau repository sur GitHub:**
   - Allez sur [GitHub](https://github.com/new)
   - Nom: `gnn-role-transition-challenge` (ou votre choix)
   - Description: "GNN Challenge: Role Transition Prediction in Temporal Networks"
   - Visibilité: **Public** (pour que les participants puissent voir le leaderboard)
   - Ne pas initialiser avec README (vous avez déjà un README)

2. **Initialiser Git localement et pousser:**
   ```bash
   cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"
   git init
   git add .
   git commit -m "Initial commit: GNN Challenge setup"
   git branch -M main
   git remote add origin https://github.com/VOTRE_USERNAME/gnn-role-transition-challenge.git
   git push -u origin main
   ```

### 2. Configurer GitHub Pages pour le Leaderboard

1. **Activer GitHub Pages:**
   - Allez dans **Settings** → **Pages**
   - Source: **Deploy from a branch**
   - Branch: `main` / `/ (root)`
   - Cliquez **Save**

2. **Le leaderboard sera accessible à:**
   ```
   https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html
   ```

### 3. Configurer les Secrets (si nécessaire)

Si vous avez besoin d'accéder à des données privées dans GitHub Actions:

1. **Settings** → **Secrets and variables** → **Actions**
2. Ajoutez des secrets si nécessaire (ex: token pour télécharger les données)

### 4. Tester le Workflow

1. **Créer une soumission de test:**
   ```bash
   # Copier un fichier de soumission
   cp submissions/sample_submission_1.csv submissions/test_team.csv
   ```

2. **Pousser et vérifier:**
   ```bash
   git add submissions/test_team.csv
   git commit -m "Add test submission"
   git push
   ```

3. **Vérifier que le workflow s'exécute:**
   - Allez dans l'onglet **Actions** de votre repository
   - Vous devriez voir le workflow "Update Leaderboard" s'exécuter

### 5. Ajouter le Lien du Leaderboard au README

Ajoutez ceci dans votre README.md (section appropriée):

```markdown
## 🏆 Leaderboard

Le leaderboard est mis à jour automatiquement à chaque soumission.

👉 **[Voir le Leaderboard](https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html)**
```

## 📝 Instructions pour les Participants

### Comment Soumettre

1. **Fork le repository**
2. **Créer votre modèle** et générer vos prédictions
3. **Placer votre fichier CSV** dans `submissions/votre_equipe.csv`
4. **Créer une Pull Request** avec votre soumission
5. Le workflow GitHub Actions évaluera automatiquement votre soumission
6. Si le score est valide, le leaderboard sera mis à jour automatiquement

### Format de Soumission

- Fichier CSV avec colonnes: `user_id`, `snapshot_id`, `predicted_role`
- Nom du fichier: `submissions/team_name.csv` (remplacer `team_name` par votre nom d'équipe)
- Le `predicted_role` doit être un entier entre 0 et 4

## 🔧 Structure des Fichiers Créés

```
.github/
└── workflows/
    ├── evaluate_submission.yml    # Évalue les soumissions via PR
    └── update_leaderboard.yml     # Met à jour le leaderboard

scripts/
├── evaluate_all_submissions.py    # Évalue toutes les soumissions
└── generate_leaderboard.py        # Génère le leaderboard HTML/JSON

update_leaderboard.py              # Script de mise à jour (alternative)
leaderboard.json                    # Données du leaderboard (généré)
leaderboard.html                    # Page HTML du leaderboard (généré)
```

## 🐛 Dépannage

### Le workflow ne s'exécute pas

1. Vérifiez que les fichiers sont dans `.github/workflows/`
2. Vérifiez la syntaxe YAML (pas d'erreurs d'indentation)
3. Vérifiez que les chemins dans les workflows sont corrects

### Le leaderboard ne se met pas à jour

1. Vérifiez les logs du workflow dans l'onglet **Actions**
2. Vérifiez que `leaderboard.json` et `leaderboard.html` sont bien commités
3. Vérifiez que GitHub Pages est activé

### Erreurs d'évaluation

1. Vérifiez que les données de test sont dans `data/private/test.parquet`
2. Vérifiez le format des fichiers CSV de soumission
3. Vérifiez que toutes les dépendances sont dans `requirements.txt`

## 📊 Personnalisation du Leaderboard

Pour modifier l'apparence du leaderboard, éditez:
- `scripts/generate_leaderboard.py` → fonction `generate_html()`

Pour modifier les métriques affichées, éditez:
- `scripts/generate_leaderboard.py` → fonction `generate_leaderboard()`

## 🔒 Sécurité

- Les données de test (`data/private/test.parquet`) ne doivent **JAMAIS** être commitées
- Elles sont dans `.gitignore`
- Les participants ne doivent avoir accès qu'aux features de test, pas aux labels

## ✅ Checklist Finale

- [ ] Repository GitHub créé
- [ ] Code poussé sur GitHub
- [ ] GitHub Pages activé
- [ ] Workflow testé avec une soumission
- [ ] Leaderboard accessible via GitHub Pages
- [ ] Lien du leaderboard ajouté au README
- [ ] Instructions de soumission ajoutées au README

---

**Besoin d'aide?** Ouvrez une issue sur le repository!

