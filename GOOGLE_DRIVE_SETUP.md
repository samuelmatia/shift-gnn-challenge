# 📤 Configuration Google Drive pour les Données Privées

Guide étape par étape pour configurer Google Drive comme stockage sécurisé pour `test.parquet`.

## 📋 Étapes

### Étape 1: Préparer le Fichier

Vérifiez que le fichier existe localement:

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"
ls -lh data/private/test.parquet
```

Vous devriez voir quelque chose comme:
```
-rw-r--r-- 1 user user 15M data/private/test.parquet
```

### Étape 2: Uploader sur Google Drive

#### Option A: Via l'Interface Web (Simple)

1. **Ouvrez Google Drive**: https://drive.google.com
2. **Connectez-vous** avec votre compte Google
3. **Créez un dossier** (optionnel mais recommandé):
   - Cliquez sur **Nouveau** → **Dossier**
   - Nommez-le: `GNN Challenge Private Data`
4. **Ouvrez le dossier** (double-clic)
5. **Uploader le fichier**:
   - Cliquez sur **Nouveau** → **Téléverser un fichier**
   - Naviguez vers: `/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge/data/private/`
   - Sélectionnez `test.parquet`
   - Attendez que l'upload se termine

#### Option B: Via Google Drive Desktop (Alternative)

Si vous avez Google Drive Desktop installé:
1. Copiez `data/private/test.parquet` dans votre dossier Google Drive
2. Le fichier sera automatiquement synchronisé

### Étape 3: Obtenir l'ID du Fichier

1. **Dans Google Drive**, faites un **clic droit** sur `test.parquet`
2. Cliquez sur **Partager** (ou **Obtenir le lien**)
3. **Configurez les permissions**:
   - Cliquez sur **Modifier** à côté de "Restreint"
   - Sélectionnez **"Toute personne avec le lien"** (ou créez un compte de service pour plus de sécurité)
   - Cliquez **Terminé**
4. **Copiez le lien de partage** qui ressemble à:
   ```
   https://drive.google.com/file/d/1ABC123xyz456DEF789ghi/view?usp=sharing
   ```
5. **Extrayez l'ID du fichier**:
   - L'ID est la partie entre `/d/` et `/view`
   - Dans l'exemple ci-dessus: `1ABC123xyz456DEF789ghi`
   - **Copiez cet ID**, vous en aurez besoin pour GitHub

### Étape 4: Tester le Téléchargement Localement

Avant de configurer GitHub, testons que le téléchargement fonctionne:

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/Pretraining/GNN Challenge"

# Installer gdown si nécessaire
pip install gdown

# Tester le téléchargement (remplacez FILE_ID par votre ID)
export GOOGLE_DRIVE_FILE_ID="1ABC123xyz456DEF789ghi"  # Votre ID ici
python scripts/download_private_data.py
```

**Résultat attendu:**
```
Downloading private test data using method: google_drive
Downloaded data/private/test.parquet
Successfully downloaded data/private/test.parquet
File size: 15.23 MB
```

Si ça fonctionne, passez à l'étape suivante. Sinon, vérifiez:
- Que l'ID du fichier est correct
- Que les permissions de partage sont correctes
- Que `gdown` est installé: `pip install gdown`

### Étape 5: Configurer GitHub Secrets

1. **Allez sur votre repository GitHub**
2. **Settings** → **Secrets and variables** → **Actions**
3. **Cliquez sur "New repository secret"**

4. **Ajoutez le premier secret:**
   - **Name**: `PRIVATE_DATA_METHOD`
   - **Value**: `google_drive`
   - Cliquez **Add secret**

5. **Ajoutez le deuxième secret:**
   - **Name**: `GOOGLE_DRIVE_FILE_ID`
   - **Value**: L'ID du fichier que vous avez copié (ex: `1ABC123xyz456DEF789ghi`)
   - Cliquez **Add secret**

### Étape 6: Vérifier la Configuration

1. **Créez une Pull Request de test**:
   ```bash
   # Créer une branche de test
   git checkout -b test-private-data
   
   # Créer une soumission de test
   cp submissions/sample_submission_1.csv submissions/test_team.csv
   
   # Commit et push
   git add submissions/test_team.csv
   git commit -m "Test private data download"
   git push origin test-private-data
   ```

2. **Créez une Pull Request** sur GitHub

3. **Vérifiez l'onglet Actions**:
   - Le workflow "Evaluate Submission" devrait s'exécuter
   - Regardez les logs de l'étape "Download private test data"
   - Vous devriez voir: `Successfully downloaded data/private/test.parquet`

4. **Vérifiez que l'évaluation fonctionne**:
   - Le workflow devrait continuer et évaluer la soumission
   - Les résultats devraient apparaître dans les commentaires de la PR

## 🔒 Sécurité (Optionnel mais Recommandé)

### Créer un Compte de Service Google (Plus Sécurisé)

Pour un accès plus sécurisé, créez un compte de service:

1. **Allez sur Google Cloud Console**: https://console.cloud.google.com
2. **Créez un nouveau projet** (ou utilisez un existant)
3. **Activez l'API Google Drive**:
   - APIs & Services → Library
   - Cherchez "Google Drive API"
   - Cliquez **Enable**
4. **Créez des identifiants**:
   - APIs & Services → Credentials
   - Cliquez **Create Credentials** → **Service Account**
   - Donnez un nom (ex: "gnn-challenge-scorer")
   - Cliquez **Create and Continue**
   - Rôle: **Editor** (ou un rôle personnalisé avec accès Drive)
   - Cliquez **Done**
5. **Créez une clé**:
   - Cliquez sur le compte de service créé
   - Onglet **Keys** → **Add Key** → **Create new key**
   - Format: **JSON**
   - Téléchargez le fichier JSON
6. **Partagez le fichier avec le compte de service**:
   - Dans Google Drive, clic droit sur `test.parquet`
   - **Partager** → Ajoutez l'email du compte de service (visible dans le JSON téléchargé)
   - Donnez les permissions **Viewer**
7. **Utilisez le token OAuth** (optionnel, plus complexe):
   - Suivez la documentation OAuth de Google
   - Stockez le token dans le secret `GOOGLE_DRIVE_ACCESS_TOKEN`

**Note**: Pour la plupart des cas, la méthode simple (lien partagé) fonctionne très bien.

## 🐛 Dépannage

### Erreur: "Failed to download private test data"

**Causes possibles:**
- L'ID du fichier est incorrect
- Les permissions de partage ne sont pas correctes
- Le fichier a été supprimé ou déplacé

**Solution:**
1. Vérifiez l'ID dans Google Drive (clic droit → Partager → copier le lien)
2. Vérifiez que le lien est accessible (ouvrez-le dans un navigateur privé)
3. Vérifiez les secrets GitHub

### Erreur: "gdown not installed"

**Solution:**
```bash
pip install gdown
```

### Le fichier téléchargé est vide

**Causes possibles:**
- Le fichier source est corrompu
- Problème de permissions

**Solution:**
1. Vérifiez que le fichier original est valide:
   ```bash
   python -c "import pandas as pd; df = pd.read_parquet('data/private/test.parquet'); print(f'Rows: {len(df)}')"
   ```
2. Ré-uploader le fichier sur Google Drive

### Le workflow GitHub Actions échoue

**Vérifications:**
1. Les secrets GitHub sont bien configurés
2. L'ID du fichier est correct (sans espaces)
3. Les logs GitHub Actions montrent l'erreur exacte

## ✅ Checklist Finale

- [ ] Fichier `test.parquet` uploadé sur Google Drive
- [ ] ID du fichier copié
- [ ] Téléchargement testé localement avec succès
- [ ] Secret `PRIVATE_DATA_METHOD` = `google_drive` configuré sur GitHub
- [ ] Secret `GOOGLE_DRIVE_FILE_ID` configuré sur GitHub
- [ ] Workflow GitHub Actions testé avec une PR
- [ ] Le fichier se télécharge correctement dans GitHub Actions
- [ ] L'évaluation fonctionne correctement

## 📝 Notes Importantes

- ⚠️ **Ne partagez JAMAIS** l'ID du fichier publiquement
- ⚠️ **Ne commitez JAMAIS** `data/private/test.parquet` dans Git
- ✅ Le fichier est déjà dans `.gitignore`
- ✅ Seul GitHub Actions peut télécharger le fichier
- ✅ Les participants n'ont pas accès aux labels

---

**Besoin d'aide?** Vérifiez les logs GitHub Actions ou consultez `SECURE_DATA_SETUP.md` pour plus de détails.

