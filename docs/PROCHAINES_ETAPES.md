# Prochaines Étapes - Configuration Finale

## ✅ Coût : **100% GRATUIT**

GitHub Actions est gratuit pour les dépôts publics avec :
- **2,000 minutes/mois** d'exécution gratuites
- Avec un polling toutes les **5 minutes** = ~8,640 exécutions/mois
- Chaque exécution prend ~1-2 minutes = **~17,280 minutes/mois maximum**
- ⚠️ **Dépassement possible** si vous avez beaucoup de soumissions

**Solution** : Si vous dépassez, vous pouvez :
- Augmenter l'intervalle à 10-15 minutes
- Utiliser un dépôt privé (2,000 minutes gratuites aussi)
- Passer à GitHub Actions payant ($0.008/minute après les 2,000 gratuites)

---

## 📋 Checklist de Configuration

### 1. **Configurer les Secrets GitHub** (5 minutes)

Allez dans votre dépôt GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

Ajoutez ces secrets :

#### a) `GOOGLE_SHEETS_ID`
- Ouvrez votre Google Sheet
- L'ID est dans l'URL : `https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit`
- Copiez `[SHEET_ID]` et collez-le dans le secret

#### b) `GOOGLE_CREDENTIALS_JSON`
- Ouvrez votre fichier `shift-gnn-challenge-16efdb090a61.json`
- Copiez **tout le contenu** du fichier JSON
- Collez-le dans le secret (même les accolades `{}`)

#### c) Secrets existants (si déjà configurés)
- `PRIVATE_DATA_METHOD` : `drive` ou `url`
- `GOOGLE_DRIVE_FILE_ID` : ID du fichier de test privé
- `GOOGLE_DRIVE_ACCESS_TOKEN` : Token d'accès (si nécessaire)
- `PRIVATE_DATA_URL` : URL alternative (si méthode = `url`)

---

### 2. **Tester le Workflow** (Optionnel - 1 minute)

⚠️ **Le workflow se déclenche automatiquement toutes les 5 minutes** - cette étape est optionnelle mais recommandée pour vérifier que tout fonctionne avant la première exécution automatique.

1. Allez dans **Actions** → **Process Google Form Submissions (Polling)**
2. Cliquez sur **"Run workflow"** → **"Run workflow"** (test manuel)
3. Vérifiez que l'exécution réussit (✅ vert)

**Note** : Même sans test manuel, le workflow s'exécutera automatiquement toutes les 5 minutes grâce au `schedule` configuré.

---

### 3. **Tester avec une Soumission** (Optionnel - 5 minutes)

1. Soumettez un CSV via votre Google Form
2. Attendez **5 minutes maximum**
3. Vérifiez que le leaderboard se met à jour automatiquement sur GitHub Pages

---

### 4. **Vérifier les Permissions Google** (Important !)

Assurez-vous que votre **Service Account** a accès :

#### a) Google Sheet
- Ouvrez votre Google Sheet
- Cliquez sur **"Partager"** (Share)
- Ajoutez l'email du Service Account (trouvable dans `shift-gnn-challenge-16efdb090a61.json` → `client_email`)
- Donnez-lui le rôle **"Éditeur"** (Editor)

#### b) Google Drive (pour les fichiers CSV soumis)
- Ouvrez le dossier Google Drive où sont stockés les fichiers CSV
- Partagez-le avec le même email du Service Account
- Donnez-lui le rôle **"Éditeur"** (Editor)

---

### 5. **Mettre à Jour le README** (Optionnel)

Ajoutez une note dans `README.md` indiquant que :
- Les soumissions sont traitées automatiquement toutes les 5 minutes
- Le leaderboard se met à jour automatiquement
- Les participants peuvent vérifier leur score après soumission

---

## 🔍 Dépannage

### Le workflow ne s'exécute pas automatiquement
- Vérifiez que le workflow est activé : **Settings** → **Actions** → **General** → **Allow all actions**
- Vérifiez le cron : doit être `*/5 * * * *`

### Erreur "Permission denied" sur Google Sheets
- Vérifiez que le Service Account a accès au Sheet (étape 4a)
- Vérifiez que `GOOGLE_CREDENTIALS_JSON` contient bien tout le JSON

### Erreur "File not found" sur Google Drive
- Vérifiez que le Service Account a accès au dossier Drive (étape 4b)
- Vérifiez que `GOOGLE_DRIVE_FILE_ID` est correct

### Le leaderboard ne se met pas à jour
- Vérifiez les logs du workflow dans **Actions**
- Vérifiez que `--push` est bien passé au script
- Vérifiez que GitHub Pages est activé : **Settings** → **Pages**

---

## 📊 Monitoring

Pour surveiller l'utilisation de GitHub Actions :
- **Settings** → **Billing** → **Actions** (pour les dépôts privés)
- Pour les dépôts publics, c'est gratuit jusqu'à 2,000 minutes/mois

---

## ✨ Résultat Final

Une fois configuré, votre système fonctionnera automatiquement :
- ✅ Polling toutes les 5 minutes
- ✅ Traitement automatique des nouvelles soumissions
- ✅ Mise à jour automatique du leaderboard
- ✅ Aucune intervention manuelle nécessaire

---

## 🆘 Besoin d'Aide ?

Si vous rencontrez des problèmes :
1. Vérifiez les logs du workflow dans **Actions**
2. Testez le script localement : `python scripts/process_google_form_submissions.py --sheets-id [ID]`
3. Vérifiez que tous les secrets sont correctement configurés
