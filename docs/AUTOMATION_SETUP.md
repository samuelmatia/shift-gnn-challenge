# 🚀 Configuration de l'Automatisation - Mise à jour Automatique du Leaderboard

## ✅ Option Recommandée : GitHub Actions avec Polling (Toutes les 5 minutes)

Cette solution vérifie automatiquement les nouvelles soumissions toutes les 5 minutes et met à jour le leaderboard.

---

## 📋 Configuration GitHub Secrets

Dans ton dépôt GitHub (Settings → Secrets and variables → Actions), ajoute :

### Secrets requis

1. **`GOOGLE_SHEETS_ID`**
   - Valeur : `1hSZlPR2GyXLbjbWurCZdBRRsAN_vB9LyiKq8XG6UTQI`
   - Description : ID de ton Google Sheet

2. **`GOOGLE_CREDENTIALS_JSON`**
   - Valeur : Contenu complet du fichier `shift-gnn-challenge-16efdb090a61.json`
   - Comment obtenir : Ouvre le fichier JSON et copie tout son contenu
   - ⚠️ Important : Copie tout le JSON (de `{` à `}`)

3. **`PRIVATE_DATA_METHOD`** (optionnel, si tu utilises des données privées)
   - Valeur : `google_drive` ou `url`
   - Description : Méthode pour télécharger les données de test privées

4. **`GOOGLE_DRIVE_FILE_ID`** (si PRIVATE_DATA_METHOD=google_drive)
   - Valeur : ID du fichier test.parquet sur Google Drive

5. **`PRIVATE_DATA_URL`** (si PRIVATE_DATA_METHOD=url)
   - Valeur : URL pour télécharger test.parquet

---

## 🔧 Activation du Workflow

1. Le workflow `.github/workflows/process_google_form_polling.yml` est déjà créé
2. Il s'exécute **automatiquement toutes les 5 minutes** (pas besoin d'activation manuelle)
3. ⚠️ **Optionnel** : Tu peux tester manuellement : **Actions** → **Process Google Form Submissions (Polling)** → **Run workflow** (recommandé pour vérifier que tout fonctionne)

---

## 🧪 Test

1. Soumets une réponse de test via ton Google Form
2. Attends 5 minutes maximum (ou déclenche manuellement le workflow)
3. Vérifie que le leaderboard est mis à jour

---

## ⚙️ Personnalisation

### Changer la fréquence de vérification

Modifie la ligne `cron` dans `.github/workflows/process_google_form_polling.yml` :

```yaml
schedule:
  - cron: '*/5 * * * *'   # Toutes les 5 minutes
  - cron: '*/10 * * * *'  # Toutes les 10 minutes (défaut)
  - cron: '*/30 * * * *'  # Toutes les 30 minutes
  - cron: '0 * * * *'     # Toutes les heures
```

### Changer le nom de la feuille

Si tu renommes ta feuille Google Sheet, modifie `--sheet-name "Feuille1"` dans le workflow.

---

## 📊 Monitoring

Pour voir les exécutions du workflow :
- Va sur **Actions** dans ton dépôt GitHub
- Clique sur **Process Google Form Submissions (Polling)**
- Tu verras l'historique des exécutions et les logs

---

## 🐛 Dépannage

### Le workflow ne se déclenche pas
- Vérifie que les secrets GitHub sont bien configurés
- Vérifie que le workflow est dans la branche `main` (ou ta branche par défaut)

### Erreur "Google credentials not found"
- Vérifie que `GOOGLE_CREDENTIALS_JSON` contient bien tout le JSON (pas juste un chemin)
- Le JSON doit commencer par `{` et finir par `}`

### Erreur "Permission denied" sur Google Sheet
- Vérifie que le compte de service a accès au Sheet (partagé avec son email)

---

## 💡 Alternative : Google Apps Script (Instantané)

Pour une mise à jour **instantanée** (dès qu'une réponse arrive), voir `docs/AUTOMATION_OPTIONS.md` section "Option 1".
