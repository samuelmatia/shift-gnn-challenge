# Correction des Problèmes du Workflow

## Problèmes Identifiés

### 1. ❌ Script manquant dans le dépôt
**Erreur** : `python: can't open file 'scripts/process_google_form_submissions.py': [Errno 2] No such file or directory`

**Cause** : Le fichier `scripts/process_google_form_submissions.py` n'était pas commité dans le dépôt GitHub.

**Solution** : ✅ Fichier ajouté et prêt à être commité.

---

### 2. ⚠️ Schedule ne se déclenche pas automatiquement

**Causes possibles** :
1. Le workflow vient d'être créé/modifié - GitHub Actions peut prendre jusqu'à **10-15 minutes** pour activer un nouveau schedule
2. GitHub Actions n'est pas activé pour les scheduled workflows dans les paramètres du dépôt
3. Le workflow doit être sur la branche par défaut (`main`)

**Solutions** :

#### Vérifier l'activation de GitHub Actions
1. Allez dans **Settings** → **Actions** → **General**
2. Vérifiez que **"Allow all actions and reusable workflows"** est sélectionné
3. Vérifiez que **"Workflow permissions"** est sur **"Read and write permissions"**

#### Vérifier que le workflow est sur la branche main
Le workflow doit être dans `.github/workflows/process_google_form_polling.yml` sur la branche `main`.

#### Attendre l'activation du schedule
GitHub Actions peut prendre jusqu'à **10-15 minutes** pour activer un nouveau schedule. Après avoir commité et poussé le workflow, attendez quelques minutes avant de vérifier.

#### Test manuel en attendant
En attendant que le schedule soit actif, vous pouvez tester manuellement :
- **Actions** → **Process Google Form Submissions (Polling)** → **Run workflow**

---

## Actions à Effectuer

### 1. Commiter et pousser les changements

```bash
git add .gitignore scripts/process_google_form_submissions.py
git commit -m "Add process_google_form_submissions.py script and update .gitignore"
git push origin main
```

### 2. Vérifier les Secrets GitHub

Assurez-vous que ces secrets sont configurés :
- `GOOGLE_SHEETS_ID`
- `GOOGLE_CREDENTIALS_JSON`
- `PRIVATE_DATA_METHOD` (optionnel)
- `GOOGLE_DRIVE_FILE_ID` (si méthode = `drive`)

### 3. Vérifier les Permissions Google

Assurez-vous que le Service Account a accès :
- Google Sheet : partagé avec l'email du Service Account (`client_email` dans le JSON)
- Google Drive : dossier des CSV partagé avec le même email

### 4. Attendre l'activation du schedule

Après avoir poussé le workflow, attendez **10-15 minutes** pour que GitHub Actions active le schedule automatique.

### 5. Vérifier que ça fonctionne

- Allez dans **Actions** → **Process Google Form Submissions (Polling)**
- Vous devriez voir des exécutions automatiques toutes les 5 minutes
- Si après 15 minutes il n'y a toujours pas d'exécution automatique, vérifiez les paramètres GitHub Actions (étape 2 ci-dessus)

---

## Note sur le Schedule GitHub Actions

GitHub Actions utilise l'**UTC** pour les schedules. Vérifiez que votre cron correspond bien à l'heure UTC souhaitée.

Le cron `*/5 * * * *` signifie : **toutes les 5 minutes**, à chaque heure.
