# 📝 Processus de Soumission - SHIFT-GNN Challenge

## 🔒 Soumissions Privées

Pour garantir la confidentialité des soumissions, le challenge utilise un **Google Form** au lieu de Pull Requests publiques.

---

## ✅ Comment Soumettre

### 1. Prépare ton fichier CSV

Crée un fichier `challenge_submission.csv` avec les colonnes suivantes :

```csv
user_id,snapshot_id,predicted_role
123,5,2
456,5,3
789,6,1
```

**Colonnes requises :**
- `user_id` : Identifiant de l'utilisateur
- `snapshot_id` : Identifiant du snapshot temporel
- `predicted_role` : Rôle prédit (entier de 0 à 4)

### 2. Soumets via Google Form

1. Ouvre le [formulaire de soumission](LINK_TO_YOUR_GOOGLE_FORM)
2. Remplis les champs :
   - **Team Name** : Ton nom d'équipe (apparaîtra sur le leaderboard)
   - **Model Type** : `human`, `llm`, ou `human+llm`
   - **Submission File** : Upload ton fichier `challenge_submission.csv`
3. Soumets le formulaire

### 3. Vérifie ton score

- Les soumissions sont traitées périodiquement (ou manuellement)
- Ton score apparaîtra sur le [leaderboard public](leaderboard.html)
- **Seuls ton nom d'équipe, tes scores et ton rang sont affichés** - ton fichier CSV reste privé

---

## 🔐 Confidentialité

- ✅ Ton fichier CSV n'est **jamais visible** par d'autres participants
- ✅ Seuls les **scores et rangs** apparaissent sur le leaderboard public
- ✅ Une seule soumission par participant (enforced par Google Form)

---

## 📊 Métriques Évaluées

- **Weighted Macro-F1** ↓ (métrique principale pour le classement)
- **Overall Macro-F1**
- **Rare Transitions F1**

Le classement suit les règles Kaggle : les scores égaux partagent le même rang.

---

## ❓ Questions ?

Si tu as des questions sur le processus de soumission, ouvre une Issue sur le dépôt GitHub.
