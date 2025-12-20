# 🧪 Guide: Tester le Leaderboard via Pull Request

Guide étape par étape pour créer une Pull Request et tester le système de leaderboard automatique.

## 📋 Prérequis

- Un compte GitHub
- Le repository `gnn-role-transition-challenge` créé sur GitHub
- Git installé sur votre machine

## 🚀 Étape par Étape

### Étape 1: Forker le Repository (Option 1 - Si vous testez depuis un autre compte)

Si vous voulez tester comme un participant (depuis un autre compte GitHub):

1. Allez sur votre repository: `https://github.com/samuelmatia/gnn-role-transition-challenge`
2. Cliquez sur le bouton **Fork** en haut à droite
3. Choisissez votre compte (ou créez un nouveau compte de test)
4. Le repository sera copié dans votre compte

**Note**: Si vous testez depuis le même compte, passez directement à l'Étape 2.

### Étape 2: Cloner le Repository (Si vous avez forké)

Si vous avez forké, clonez votre fork:

```bash
# Remplacez VOTRE_USERNAME par votre nom d'utilisateur GitHub
git clone https://github.com/VOTRE_USERNAME/gnn-role-transition-challenge.git
cd gnn-role-transition-challenge
```

**Si vous testez depuis le même compte**, vous travaillez déjà dans le bon répertoire.

### Étape 3: Créer une Branche pour la Soumission de Test

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"

# Créer une nouvelle branche
git checkout -b test-submission

# Vérifier que vous êtes sur la bonne branche
git branch
# Vous devriez voir * test-submission
```

### Étape 4: Créer une Soumission de Test

```bash
# Copier un fichier de soumission exemple
cp submissions/sample_submission_1.csv submissions/test_team_awesome.csv

# Vérifier que le fichier existe
ls -lh submissions/test_team_awesome.csv
```

### Étape 5: Commiter la Soumission

```bash
# Ajouter le fichier
git add submissions/test_team_awesome.csv

# Vérifier ce qui va être commité
git status

# Commiter
git commit -m "Add test submission: test_team_awesome"
```

### Étape 6: Pousser la Branche sur GitHub

```bash
# Pousser la branche (la première fois)
git push origin test-submission

# Si vous avez forké, poussez vers votre fork:
# git push origin test-submission
```

**Note**: Si c'est la première fois que vous poussez cette branche, Git vous donnera peut-être une commande à exécuter. Copiez-collez la commande suggérée.

### Étape 7: Créer la Pull Request sur GitHub

1. **Allez sur votre repository GitHub**:
   - Si vous avez forké: `https://github.com/VOTRE_USERNAME/gnn-role-transition-challenge`
   - Sinon: `https://github.com/samuelmatia/gnn-role-transition-challenge`

2. **Vous verrez une bannière jaune** en haut de la page qui dit:
   ```
   test-submission had recent pushes
   [Compare & pull request]
   ```
   Cliquez sur **"Compare & pull request"**

   **OU** cliquez sur l'onglet **"Pull requests"** puis sur **"New pull request"**

3. **Remplissez le formulaire de Pull Request**:
   - **Base**: `main` (la branche principale)
   - **Compare**: `test-submission` (votre branche)
   - **Title**: `Test submission: test_team_awesome`
   - **Description**: 
     ```
     This is a test submission to verify the leaderboard system works correctly.
     ```

4. **Cliquez sur "Create pull request"**

### Étape 8: Observer le Workflow GitHub Actions

1. **Une fois la PR créée**, allez dans l'onglet **"Actions"** de votre repository
2. **Vous devriez voir** le workflow "Evaluate Submission" s'exécuter
3. **Cliquez sur le workflow** pour voir les détails:
   - Il va télécharger les données privées
   - Évaluer votre soumission
   - Mettre à jour le leaderboard
   - Poster un commentaire sur la PR

4. **Attendez que le workflow se termine** (peut prendre 2-5 minutes)

### Étape 9: Vérifier les Résultats

1. **Retournez sur la Pull Request** (onglet "Pull requests")
2. **Ouvrez votre PR** (`test-submission`)
3. **Regardez les commentaires**:
   - Un bot GitHub Actions devrait avoir posté un commentaire avec les résultats
   - Vous verrez les scores: Weighted Macro-F1, Overall Macro-F1, etc.

4. **Vérifiez le leaderboard**:
   - Allez sur: `https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html`
   - Votre équipe "test_team_awesome" devrait apparaître (si le score est valide)

### Étape 10: Fusionner la PR (Optionnel)

Si tout fonctionne bien, vous pouvez fusionner la PR:

1. **Dans la PR**, cliquez sur **"Merge pull request"**
2. **Confirmez** en cliquant sur **"Confirm merge"**
3. **Optionnel**: Supprimez la branche après fusion

## 🔍 Vérifications

### Vérifier que le Workflow s'est Exécuté

1. Onglet **Actions** → Cherchez "Evaluate Submission"
2. Vérifiez qu'il est marqué **✓** (succès) et non **✗** (échec)

### Vérifier les Commentaires sur la PR

1. Onglet **Pull requests** → Ouvrez votre PR
2. Scroll vers le bas pour voir les commentaires
3. Vous devriez voir un commentaire avec les résultats d'évaluation

### Vérifier le Leaderboard

1. Visitez: `https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html`
2. Votre équipe devrait apparaître avec son score

## 🐛 Dépannage

### Le workflow ne s'exécute pas

**Problème**: Le workflow "Evaluate Submission" ne se déclenche pas

**Solutions**:
1. Vérifiez que le fichier CSV est bien dans `submissions/`
2. Vérifiez que le nom ne contient pas "sample" (les fichiers sample sont ignorés)
3. Vérifiez les logs dans l'onglet Actions

### Erreur: "Failed to download private test data"

**Problème**: Le workflow ne peut pas télécharger les données de test

**Solutions**:
1. Vérifiez que les secrets GitHub sont configurés:
   - `PRIVATE_DATA_METHOD` = `google_drive`
   - `GOOGLE_DRIVE_FILE_ID` = votre ID
2. Vérifiez les logs du workflow pour plus de détails

### Le leaderboard ne se met pas à jour

**Problème**: La PR est évaluée mais le leaderboard ne change pas

**Solutions**:
1. Vérifiez que le workflow s'est terminé avec succès
2. Attendez quelques minutes (GitHub Pages peut prendre du temps)
3. Videz le cache de votre navigateur
4. Vérifiez que `leaderboard.json` et `leaderboard.html` ont été mis à jour dans le repository

## 📝 Commandes Récapitulatives

Voici toutes les commandes en une fois (pour référence):

```bash
# 1. Créer une branche
git checkout -b test-submission

# 2. Créer une soumission de test
cp submissions/sample_submission_1.csv submissions/test_team_awesome.csv

# 3. Commiter
git add submissions/test_team_awesome.csv
git commit -m "Add test submission: test_team_awesome"

# 4. Pousser
git push origin test-submission

# Ensuite, allez sur GitHub pour créer la PR
```

## ✅ Checklist de Test

- [ ] Branche créée (`test-submission`)
- [ ] Fichier de soumission créé dans `submissions/`
- [ ] Fichier commité et poussé
- [ ] Pull Request créée sur GitHub
- [ ] Workflow "Evaluate Submission" s'exécute
- [ ] Workflow se termine avec succès
- [ ] Commentaire avec résultats posté sur la PR
- [ ] Leaderboard mis à jour avec la nouvelle soumission

## 🎉 C'est Tout!

Une fois que vous avez réussi à créer une PR et que le workflow fonctionne, vous savez comment tester le système. Les participants pourront faire exactement la même chose!

---

**Besoin d'aide?** Vérifiez les logs dans l'onglet Actions ou consultez les autres guides de dépannage.

