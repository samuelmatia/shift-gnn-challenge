# Guide de Test pour les Pull Requests

Ce document explique comment tester le système de soumission via Pull Request et la mise à jour automatique du leaderboard.

## 🔍 Vérifications Préalables

### 1. Vérifier la Structure du Projet

Assurez-vous que les fichiers suivants existent :
- ✅ `.github/workflows/evaluate_submission.yml` - Workflow GitHub Actions
- ✅ `scoring_script.py` - Script d'évaluation
- ✅ `scripts/extract_scores.py` - Extraction des scores
- ✅ `scripts/update_leaderboard_from_scores.py` - Mise à jour du leaderboard
- ✅ `scripts/generate_leaderboard.py` - Génération HTML
- ✅ `scripts/download_private_data.py` - Téléchargement des données privées
- ✅ `data/private/test.parquet` - Fichier de test (ou configuré via secrets)

### 2. Vérifier les Secrets GitHub

Le workflow nécessite des secrets GitHub pour télécharger les données privées. Vérifiez dans **Settings > Secrets and variables > Actions** :

- `PRIVATE_DATA_METHOD` (optionnel, défaut: 'url')
- `PRIVATE_DATA_URL` (si méthode = 'url')
- `PRIVATE_DATA_TOKEN` (optionnel, si authentification requise)
- Ou d'autres secrets selon la méthode choisie (Google Drive, S3, etc.)

### 3. Vérifier les Permissions du Workflow

Le workflow doit avoir les permissions suivantes :
- ✅ `contents: write` - Pour commit/push
- ✅ `pull-requests: write` - Pour commenter les PRs

Ces permissions sont déjà configurées dans le workflow.

## 📝 Étapes pour Tester un Pull Request

### Étape 1 : Préparer un Fichier de Soumission de Test

1. **Créer une branche de test** :
   ```bash
   git checkout -b test-submission-pr
   ```

2. **Créer un fichier de soumission de test** :
   - Nom du fichier : `submissions/test_team.csv`
   - Format requis :
     ```csv
     user_id,snapshot_id,predicted_role
     123,5,2
     456,5,3
     789,6,1
     ```
   - ⚠️ **Important** : Le fichier doit contenir les mêmes `user_id` et `snapshot_id` que dans `data/private/test.parquet`

3. **Optionnel : Utiliser un fichier existant pour tester** :
   ```bash
   # Copier un fichier existant avec un nouveau nom
   cp submissions/team_sam_trad_ML_RandomForest.csv submissions/test_team_pr.csv
   ```

### Étape 2 : Tester Localement (Optionnel mais Recommandé)

Avant de créer le PR, testez localement :

```bash
# 1. Tester le scoring
python scoring_script.py submissions/test_team_pr.csv

# 2. Tester l'extraction des scores
python scripts/extract_scores.py test_team_pr

# 3. Vérifier que le fichier de test existe
ls -la data/private/test.parquet
```

### Étape 3 : Créer le Pull Request

1. **Commit et push la branche** :
   ```bash
   git add submissions/test_team_pr.csv
   git commit -m "Test: Add submission file for PR testing"
   git push origin test-submission-pr
   ```

2. **Créer le Pull Request sur GitHub** :
   - Aller sur GitHub
   - Cliquer sur "New Pull Request"
   - Sélectionner `test-submission-pr` → `main`
   - Titre : "Test: Submission via PR"
   - Description : "Test du système de soumission via Pull Request"
   - Cliquer sur "Create Pull Request"

### Étape 4 : Vérifier l'Exécution du Workflow

1. **Surveiller le workflow** :
   - Aller dans l'onglet "Actions" sur GitHub
   - Le workflow "Evaluate Submission" devrait se déclencher automatiquement
   - Cliquer sur le workflow en cours pour voir les logs

2. **Vérifier les étapes** :
   - ✅ Checkout repository
   - ✅ Set up Python
   - ✅ Install dependencies
   - ✅ Download private test data
   - ✅ Find submission files (doit trouver `submissions/test_team_pr.csv`)
   - ✅ Evaluate submissions
   - ✅ Update leaderboard
   - ✅ Generate HTML leaderboard
   - ✅ Comment PR with results
   - ✅ Upload leaderboard artifacts (pour PR)

### Étape 5 : Vérifier les Résultats

1. **Commentaire sur le PR** :
   - Le workflow devrait ajouter un commentaire sur le PR avec les résultats
   - Vérifier que les scores sont affichés correctement

2. **Artifacts** :
   - Dans l'onglet "Actions", télécharger l'artifact "leaderboard-update"
   - Vérifier que `leaderboard.json` et `leaderboard.html` sont générés
   - Vérifier que votre équipe apparaît dans le leaderboard avec les bons scores

3. **Fichiers dans le PR** :
   - Vérifier que le fichier `submissions/test_team_pr.csv` est bien présent dans le PR
   - Vérifier que le fichier est dans le bon format

### Étape 6 : Merger le PR (Test Complet)

1. **Merger le PR** :
   - Cliquer sur "Merge pull request"
   - Confirmer le merge

2. **Vérifier après le merge** :
   - Le workflow devrait se déclencher à nouveau sur `push` vers `main`
   - Cette fois, le commit et push du leaderboard devrait fonctionner
   - Vérifier que `leaderboard.json` et `leaderboard.html` sont mis à jour dans la branche `main`
   - Vérifier que le leaderboard en ligne est mis à jour (si GitHub Pages est configuré)

## 🐛 Dépannage

### Problème : Le workflow ne se déclenche pas

**Solutions** :
- Vérifier que le fichier CSV est bien dans `submissions/`
- Vérifier que le workflow est dans `.github/workflows/`
- Vérifier les permissions du repository (Settings > Actions > General)

### Problème : "No submission files found"

**Solutions** :
- Vérifier que le fichier est bien un `.csv`
- Vérifier que le fichier est dans `submissions/`
- Vérifier les logs de l'étape "Find submission files"

### Problème : "Failed to download private test data"

**Solutions** :
- Vérifier que les secrets GitHub sont configurés
- Vérifier que `data/private/test.parquet` existe localement
- Pour les tests, vous pouvez modifier temporairement le workflow pour utiliser un fichier local

### Problème : Le leaderboard n'est pas mis à jour après le merge

**Solutions** :
- Vérifier que le workflow s'est exécuté après le merge
- Vérifier les logs du workflow
- Vérifier que le commit a été fait (git log)
- Vérifier que GitHub Pages est configuré (si applicable)

### Problème : Erreur dans l'évaluation

**Solutions** :
- Vérifier le format du fichier CSV
- Vérifier que les colonnes sont : `user_id`, `snapshot_id`, `predicted_role`
- Vérifier que les valeurs de `predicted_role` sont entre 0 et 4
- Vérifier que les `user_id` et `snapshot_id` correspondent à ceux dans `test.parquet`

## ✅ Checklist de Vérification

Avant de tester, vérifiez :

- [ ] Le workflow `.github/workflows/evaluate_submission.yml` existe
- [ ] Les scripts Python sont présents et fonctionnels
- [ ] Les secrets GitHub sont configurés (si nécessaire)
- [ ] Le fichier `data/private/test.parquet` existe ou est accessible
- [ ] Vous avez les permissions pour créer des PRs
- [ ] Le repository a GitHub Actions activé

## 📊 Résultats Attendus

Après un PR réussi, vous devriez voir :

1. ✅ Un commentaire sur le PR avec les scores
2. ✅ Un artifact "leaderboard-update" avec les fichiers générés
3. ✅ Le workflow terminé avec succès (toutes les étapes vertes)
4. ✅ Après le merge : `leaderboard.json` et `leaderboard.html` mis à jour dans `main`
5. ✅ Votre équipe apparaît dans le leaderboard avec les bons scores

## 🔄 Workflow Complet

```
PR créé avec fichier CSV
    ↓
Workflow se déclenche
    ↓
Télécharge les données privées
    ↓
Trouve les fichiers CSV modifiés
    ↓
Évalue chaque soumission
    ↓
Extrait les scores
    ↓
Met à jour le leaderboard
    ↓
Génère le HTML
    ↓
Commente le PR avec les résultats
    ↓
Upload les artifacts (pour PR)
    ↓
[Après merge] Commit et push le leaderboard
```

## 📝 Notes Importantes

1. **Pour les PRs** : Le leaderboard n'est pas commité directement dans la branche du PR (pour éviter les problèmes de permissions). Il est disponible en artifact.

2. **Après le merge** : Le workflow se déclenche à nouveau et commit/push le leaderboard dans `main`.

3. **Format du fichier** : Le nom du fichier détermine le nom de l'équipe (sans l'extension `.csv`).

4. **Scores** : Seul le meilleur score par équipe est conservé dans le leaderboard.

