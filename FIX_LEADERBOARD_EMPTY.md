# 🔧 Fix: Leaderboard Vide - Aucune Soumission

## Problème

Le workflow "Update Leaderboard" réussit mais aucune soumission n'apparaît dans le leaderboard.

## Causes

1. **Les fichiers "sample" sont ignorés** : Le script `evaluate_all_submissions.py` ignore tous les fichiers contenant "sample" dans le nom
2. **Les autres fichiers CSV sont ignorés par Git** : Le `.gitignore` exclut tous les CSV sauf les samples
3. **Aucune vraie soumission n'est dans le repository**

## Solutions

### Solution 1: Créer une Vraie Soumission de Test

Pour tester le système, créez une soumission qui ne contient pas "sample" dans le nom :

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"

# Créer une soumission de test (sans "sample" dans le nom)
cp submissions/sample_submission_1.csv submissions/test_team.csv

# Ajouter au repository (forcer l'ajout malgré .gitignore)
git add -f submissions/test_team.csv

# Commiter
git commit -m "Add test submission: test_team"

# Pousser
git push origin main
```

**Important** : Utilisez `git add -f` pour forcer l'ajout malgré `.gitignore`.

### Solution 2: Modifier .gitignore pour Autoriser les Soumissions

Si vous voulez permettre certaines soumissions dans Git :

```bash
# Modifier .gitignore pour autoriser les fichiers de test
# Ajoutez cette ligne :
!submissions/test_*.csv
```

Puis :
```bash
git add submissions/test_team.csv
git commit -m "Add test submission"
git push origin main
```

### Solution 3: Utiliser le Workflow evaluate_submission

J'ai modifié le workflow `evaluate_submission.yml` pour qu'il se déclenche aussi sur les pushes vers `main`. 

Maintenant, quand vous poussez un fichier CSV dans `submissions/` :
- ✅ Le workflow `evaluate_submission` s'exécutera
- ✅ Le workflow `update_leaderboard` s'exécutera aussi

## Vérification

Après avoir créé une soumission de test :

1. **Vérifiez que le fichier est dans Git** :
   ```bash
   git ls-files submissions/
   ```
   Vous devriez voir `submissions/test_team.csv`

2. **Vérifiez les workflows** :
   - Allez sur GitHub → Onglet **Actions**
   - Vous devriez voir "Evaluate Submission" s'exécuter
   - Puis "Update Leaderboard" s'exécuter

3. **Vérifiez le leaderboard** :
   - Visitez: `https://samuelmatia.github.io/gnn-role-transition-challenge/leaderboard.html`
   - Votre équipe devrait apparaître

## Workflows Actifs

Maintenant, les deux workflows se déclenchent sur push vers `main` :

| Workflow | Déclencheur |
|----------|-------------|
| `evaluate_submission.yml` | ✅ Pull Request avec CSV<br>✅ Push sur main avec CSV |
| `update_leaderboard.yml` | ✅ Push sur main avec CSV |

## Note Importante

Les fichiers "sample" sont intentionnellement ignorés pour éviter d'évaluer les exemples. Pour tester, créez une soumission avec un nom qui ne contient pas "sample".

