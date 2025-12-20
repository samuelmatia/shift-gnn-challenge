# ✅ Checklist de Déploiement GitHub

## 📋 Étapes à Suivre

### 1. Préparation Locale ✅

- [x] Code prêt et testé
- [x] `.gitignore` configuré
- [x] Workflows GitHub Actions créés
- [x] Scripts de leaderboard créés
- [x] Documentation mise à jour

### 2. Créer le Repository GitHub

```bash
# 1. Créer un nouveau repository sur GitHub.com
#    - Nom: gnn-role-transition-challenge (ou votre choix)
#    - Public (pour GitHub Pages)
#    - Ne PAS initialiser avec README

# 2. Dans votre terminal, exécutez:
cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"

# 3. Initialiser Git (si pas déjà fait)
git init
git add .
git commit -m "Initial commit: GNN Challenge with auto-leaderboard"

# 4. Ajouter le remote et pousser
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/gnn-role-transition-challenge.git
git push -u origin main
```

**Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur GitHub.**

### 3. Configurer GitHub Pages

1. Allez dans votre repository → **Settings**
2. Dans le menu de gauche, cliquez sur **Pages**
3. Sous **Source**, sélectionnez:
   - Branch: `main`
   - Folder: `/ (root)`
4. Cliquez **Save**

Le leaderboard sera accessible à:
```
https://VOTRE_USERNAME.github.io/gnn-role-transition-challenge/leaderboard.html
```

### 4. Tester le Système

#### Test 1: Vérifier que les fichiers sont bien poussés

```bash
# Vérifier que tous les fichiers sont présents
git ls-files | grep -E "(workflow|script|leaderboard)"
```

Vous devriez voir:
- `.github/workflows/evaluate_submission.yml`
- `.github/workflows/update_leaderboard.yml`
- `scripts/evaluate_all_submissions.py`
- `scripts/generate_leaderboard.py`
- `update_leaderboard.py`

#### Test 2: Créer une soumission de test

```bash
# Créer une soumission de test
cp submissions/sample_submission_1.csv submissions/test_team.csv

# Pousser
git add submissions/test_team.csv
git commit -m "Add test submission"
git push
```

#### Test 3: Vérifier le workflow

1. Allez dans l'onglet **Actions** de votre repository
2. Vous devriez voir le workflow "Evaluate Submission" s'exécuter
3. Vérifiez qu'il se termine avec succès
4. Vérifiez que `leaderboard.json` et `leaderboard.html` sont créés/mis à jour

### 5. Ajouter le Lien du Leaderboard au README

Ouvrez `README.md` et ajoutez/modifiez la section leaderboard avec votre URL:

```markdown
## 🏆 Leaderboard

👉 **[View Live Leaderboard](https://VOTRE_USERNAME.github.io/gnn-role-transition-challenge/leaderboard.html)**
```

Puis:
```bash
git add README.md
git commit -m "Add leaderboard link"
git push
```

### 6. Configuration Avancée (Optionnel)

#### Si vous avez besoin de données privées dans GitHub Actions

1. **Settings** → **Secrets and variables** → **Actions**
2. Cliquez **New repository secret**
3. Ajoutez vos secrets (ex: token pour télécharger les données)

#### Personnaliser le leaderboard

Pour modifier l'apparence:
- Éditez `scripts/generate_leaderboard.py` → fonction `generate_html()`

Pour modifier les métriques:
- Éditez `scripts/generate_leaderboard.py` → fonction `generate_leaderboard()`

### 8. Instructions pour les Participants

Les participants doivent:

1. **Fork** le repository
2. **Télécharger** les données dans `data/processed/`
3. **Créer** leur modèle
4. **Générer** leurs prédictions
5. **Placer** leur fichier CSV dans `submissions/team_name.csv`
6. **Créer une Pull Request**

Le workflow GitHub Actions:
- ✅ Évalue automatiquement la soumission
- ✅ Poste les résultats en commentaire sur la PR
- ✅ Met à jour le leaderboard si le score est valide
- ✅ Régénère la page HTML du leaderboard

## 🔍 Vérification Finale

Avant de publier, vérifiez:

- [ ] Repository GitHub créé et code poussé
- [ ] GitHub Pages activé
- [ ] **Données privées configurées** (secrets GitHub + upload sécurisé)
- [ ] **Téléchargement des données privées testé** (localement et dans GitHub Actions)
- [ ] Workflow testé avec une soumission
- [ ] Leaderboard accessible via GitHub Pages
- [ ] Lien du leaderboard dans le README
- [ ] Instructions de soumission claires
- [ ] `.gitignore` exclut bien les données sensibles
- [ ] **Le fichier `data/private/test.parquet` n'est PAS dans le repository**

## 🐛 Dépannage

### Le workflow ne s'exécute pas

**Problème**: Le workflow ne se déclenche pas sur les PRs

**Solution**:
1. Vérifiez la syntaxe YAML (pas d'erreurs d'indentation)
2. Vérifiez que le fichier est dans `.github/workflows/`
3. Vérifiez les logs dans l'onglet **Actions**

### Erreur "No submission files found"

**Problème**: Le workflow ne trouve pas les fichiers CSV

**Solution**:
1. Vérifiez que les fichiers sont bien dans `submissions/`
2. Vérifiez que les noms ne contiennent pas "sample"
3. Vérifiez les chemins dans le workflow

### Le leaderboard ne se met pas à jour

**Problème**: Le leaderboard reste vide ou ne se met pas à jour

**Solution**:
1. Vérifiez que `data/private/test.parquet` existe (localement, pas sur GitHub)
2. Vérifiez les logs du workflow pour les erreurs
3. Vérifiez que `leaderboard.json` et `leaderboard.html` sont commités
4. Vérifiez que GitHub Pages est activé

### Erreur d'évaluation

**Problème**: Le scoring échoue

**Solution**:
1. Vérifiez le format du CSV (colonnes: user_id, snapshot_id, predicted_role)
2. Vérifiez que toutes les dépendances sont dans `requirements.txt`
3. Vérifiez les logs détaillés dans l'onglet **Actions**

## 📞 Support

Si vous rencontrez des problèmes:
1. Vérifiez les logs dans l'onglet **Actions**
2. Consultez `GITHUB_SETUP.md` pour plus de détails
3. Ouvrez une issue sur le repository

---

**🎉 Une fois toutes les étapes complétées, votre challenge est prêt!**

Les participants pourront soumettre leurs solutions et le leaderboard se mettra à jour automatiquement.

