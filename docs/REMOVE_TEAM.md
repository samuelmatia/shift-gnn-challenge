# Comment supprimer une équipe du leaderboard

Il existe deux méthodes pour supprimer une équipe du leaderboard :

## Méthode 1 : Via GitHub Actions (Recommandé)

### Option A : Via Workflow Dispatch (Interface GitHub)

1. Allez dans l'onglet **Actions** de votre repository GitHub
2. Sélectionnez le workflow **"Remove Team from Leaderboard"**
3. Cliquez sur **"Run workflow"**
4. Entrez le nom de l'équipe à supprimer (ex: `team_sam_trad_ML_RandomForest`)
5. Cliquez sur **"Run workflow"**

Le workflow va :
- ✅ Supprimer l'équipe du leaderboard
- ✅ Régénérer le HTML
- ✅ Commiter et pousser les changements sur `main`
- ✅ Déclencher le déploiement GitHub Pages

### Option B : Via Issue GitHub

1. Créez une nouvelle **Issue** sur GitHub
2. Ajoutez le label **`remove-team`**
3. Dans le titre ou le corps de l'issue, mentionnez l'équipe à supprimer :
   - Format recommandé : `Remove team: team_name`
   - Ou simplement mentionnez le nom de l'équipe dans le corps

Le workflow va automatiquement :
- ✅ Détecter l'issue avec le label `remove-team`
- ✅ Extraire le nom de l'équipe
- ✅ Supprimer l'équipe du leaderboard
- ✅ Commenter sur l'issue pour confirmer
- ✅ Fermer l'issue automatiquement

## Méthode 2 : Via ligne de commande (Local)

Si vous travaillez localement sur la branche `main` :

```bash
python3 remove_team.py <team_name>
```

Exemple :
```bash
python3 remove_team.py team_sam_trad_ML_RandomForest
```

Puis commitez et poussez les changements :
```bash
git add leaderboard.json leaderboard.html
git commit -m "Remove team from leaderboard"
git push origin main
```

## Notes importantes

- ⚠️ La suppression se fait toujours sur la branche `main`
- ✅ Le leaderboard HTML est automatiquement régénéré
- ✅ GitHub Pages se met à jour automatiquement après le push
- 📊 Vous pouvez vérifier les équipes disponibles dans `leaderboard.json`

## Vérification

Après suppression, vérifiez que :
1. L'équipe n'apparaît plus dans `leaderboard.json`
2. Le fichier `leaderboard.html` a été mis à jour
3. Les changements ont été poussés sur `main`
4. Le site GitHub Pages reflète les changements

