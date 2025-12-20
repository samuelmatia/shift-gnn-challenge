# 🔒 Configuration des Données Privées (Test Labels)

Le fichier `data/private/test.parquet` contient les labels de test et **ne doit JAMAIS** être accessible aux participants. Ce guide explique comment configurer un accès sécurisé pour GitHub Actions.

## ⚠️ Important

- ❌ **NE JAMAIS** commiter `data/private/test.parquet` dans le repository
- ✅ Le fichier est déjà dans `.gitignore`
- ✅ GitHub Actions télécharge le fichier depuis un emplacement sécurisé
- ✅ Seuls les organisateurs ont accès aux données privées

## 📋 Options de Stockage Sécurisé

### Option 1: Google Drive (Recommandé pour débuter)

#### Étape 1: Uploader le fichier sur Google Drive

1. Allez sur [Google Drive](https://drive.google.com)
2. Créez un dossier "GNN Challenge Private Data"
3. Uploader `data/private/test.parquet`
4. Clic droit sur le fichier → **Partager** → **Obtenir le lien**
5. Configurez l'accès: **"Toute personne avec le lien"** (ou créez un compte de service)
6. Copiez l'ID du fichier depuis l'URL:
   ```
   https://drive.google.com/file/d/FILE_ID_HERE/view
                                    ^^^^^^^^^^^^^^
                                    C'est l'ID dont vous avez besoin
   ```

#### Étape 2: Configurer GitHub Secrets

1. Allez dans votre repository GitHub → **Settings** → **Secrets and variables** → **Actions**
2. Cliquez **New repository secret**
3. Ajoutez les secrets suivants:

   - **Nom**: `PRIVATE_DATA_METHOD`
     **Valeur**: `google_drive`

   - **Nom**: `GOOGLE_DRIVE_FILE_ID`
     **Valeur**: L'ID du fichier copié précédemment

   - **Nom**: `GOOGLE_DRIVE_ACCESS_TOKEN` (Optionnel, pour accès privé)
     **Valeur**: Token d'accès OAuth (voir ci-dessous)

#### Optionnel: Créer un Token d'Accès OAuth

Pour un accès plus sécurisé:

1. Allez sur [Google Cloud Console](https://console.cloud.google.com)
2. Créez un projet
3. Activez l'API Google Drive
4. Créez des identifiants OAuth 2.0
5. Utilisez le token d'accès dans le secret

### Option 2: URL Privée (Simple)

#### Étape 1: Héberger le fichier

Hébergez `test.parquet` sur:
- Un serveur privé avec authentification
- Un service cloud (Dropbox, OneDrive, etc.)
- Un serveur web avec protection par token

#### Étape 2: Configurer GitHub Secrets

1. **Settings** → **Secrets and variables** → **Actions**
2. Ajoutez:

   - **Nom**: `PRIVATE_DATA_METHOD`
     **Valeur**: `url`

   - **Nom**: `PRIVATE_DATA_URL`
     **Valeur**: URL complète vers le fichier (ex: `https://votre-serveur.com/data/test.parquet`)

   - **Nom**: `PRIVATE_DATA_TOKEN` (Optionnel)
     **Valeur**: Token d'authentification si nécessaire

### Option 3: Amazon S3 (Production)

#### Étape 1: Uploader sur S3

```bash
aws s3 cp data/private/test.parquet s3://votre-bucket/data/private/test.parquet
```

#### Étape 2: Configurer GitHub Secrets

1. **Settings** → **Secrets and variables** → **Actions**
2. Ajoutez:

   - **Nom**: `PRIVATE_DATA_METHOD`
     **Valeur**: `s3`

   - **Nom**: `S3_BUCKET`
     **Valeur**: Nom de votre bucket

   - **Nom**: `S3_KEY`
     **Valeur**: `data/private/test.parquet` (ou votre chemin)

   - **Nom**: `AWS_ACCESS_KEY_ID`
     **Valeur**: Votre clé d'accès AWS

   - **Nom**: `AWS_SECRET_ACCESS_KEY`
     **Valeur**: Votre clé secrète AWS

## ✅ Vérification

### Tester Localement

```bash
# Tester avec Google Drive
export PRIVATE_DATA_METHOD=google_drive
export GOOGLE_DRIVE_FILE_ID=votre_file_id
python scripts/download_private_data.py

# Tester avec URL
export PRIVATE_DATA_METHOD=url
export PRIVATE_DATA_URL=https://votre-url.com/test.parquet
python scripts/download_private_data.py

# Tester avec S3
export PRIVATE_DATA_METHOD=s3
export S3_BUCKET=votre-bucket
export AWS_ACCESS_KEY_ID=votre_key
export AWS_SECRET_ACCESS_KEY=votre_secret
python scripts/download_private_data.py
```

### Tester dans GitHub Actions

1. Créez une Pull Request de test
2. Vérifiez dans l'onglet **Actions** que le workflow s'exécute
3. Vérifiez les logs pour confirmer que le téléchargement fonctionne

## 🔐 Sécurité Avancée

### Restreindre l'Accès au Fichier

1. **Google Drive**: Utilisez un compte de service avec accès limité
2. **URL**: Utilisez un token d'authentification qui expire
3. **S3**: Utilisez des politiques IAM restrictives

### Rotation des Secrets

- Changez régulièrement les tokens d'accès
- Utilisez des tokens avec expiration
- Surveillez l'utilisation dans les logs GitHub Actions

### Monitoring

- Vérifiez régulièrement les logs GitHub Actions
- Surveillez les accès aux données privées
- Alertez en cas d'accès suspect

## 🐛 Dépannage

### Erreur: "Failed to download private test data"

**Causes possibles:**
- Secret GitHub mal configuré
- URL/ID de fichier incorrect
- Token expiré ou invalide
- Problème de permissions

**Solution:**
1. Vérifiez que tous les secrets sont correctement configurés
2. Testez le téléchargement localement
3. Vérifiez les logs GitHub Actions pour plus de détails

### Erreur: "File is empty or doesn't exist"

**Causes possibles:**
- Le téléchargement a échoué silencieusement
- Le fichier source est corrompu
- Problème de permissions d'écriture

**Solution:**
1. Vérifiez que le fichier source existe et est valide
2. Vérifiez les permissions du répertoire `data/private/`
3. Testez manuellement le téléchargement

## 📝 Checklist

- [ ] Fichier `test.parquet` uploadé sur un service sécurisé
- [ ] GitHub Secrets configurés
- [ ] Méthode de téléchargement testée localement
- [ ] Workflow GitHub Actions testé avec une PR
- [ ] Vérification que le fichier n'est PAS dans le repository
- [ ] Documentation mise à jour pour les organisateurs

---

**⚠️ Rappel**: Ne partagez JAMAIS les secrets GitHub ou les URLs privées avec les participants!

