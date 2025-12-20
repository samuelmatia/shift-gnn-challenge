# 🔍 Debug: Workflow Evaluate Submission qui Échoue

## Problèmes Identifiés

D'après l'historique GitHub Actions, le workflow `evaluate_submission` se déclenche mais échoue. Voici les causes probables :

### 1. Données Privées Non Configurées

Le workflow échoue probablement à l'étape "Download private test data" car les secrets GitHub ne sont pas configurés.

**Solution** : Configurez les secrets GitHub (voir `GOOGLE_DRIVE_SETUP.md`)

### 2. Aucun Fichier de Soumission Trouvé

Le workflow peut échouer si aucun fichier CSV n'est trouvé dans `submissions/`.

**Solution** : Créez une soumission de test :
```bash
cp submissions/sample_submission_1.csv submissions/test_team.csv
git add -f submissions/test_team.csv
git commit -m "Add test submission"
git push origin main
```

### 3. Problème avec git diff sur Push Direct

Sur un push direct vers `main`, `git diff` ne fonctionne pas correctement.

**Solution** : J'ai corrigé le workflow pour utiliser `git ls-files` sur les pushes directs.

## Vérifications à Faire

### Vérifier les Secrets GitHub

1. Allez sur votre repository → **Settings** → **Secrets and variables** → **Actions**
2. Vérifiez que ces secrets existent :
   - `PRIVATE_DATA_METHOD` = `google_drive`
   - `GOOGLE_DRIVE_FILE_ID` = votre ID de fichier

### Vérifier les Fichiers de Soumission

```bash
# Vérifier les fichiers dans Git
git ls-files submissions/

# Vérifier les fichiers locaux
ls -la submissions/*.csv
```

### Vérifier les Logs du Workflow

1. Allez sur GitHub → **Actions**
2. Cliquez sur un workflow qui a échoué
3. Regardez les logs pour voir à quelle étape il échoue

## Corrections Appliquées

J'ai corrigé :
1. ✅ La détection des fichiers pour les pushes directs vers `main`
2. ✅ Meilleure gestion d'erreur pour le téléchargement des données privées
3. ✅ Messages d'erreur plus clairs

## Prochaines Étapes

1. **Configurer les secrets GitHub** (si pas déjà fait)
2. **Créer une soumission de test** :
   ```bash
   cp submissions/sample_submission_1.csv submissions/test_team.csv
   git add -f submissions/test_team.csv
   git commit -m "Add test submission: test_team"
   git push origin main
   ```
3. **Vérifier les logs** dans l'onglet Actions pour voir où ça échoue exactement

