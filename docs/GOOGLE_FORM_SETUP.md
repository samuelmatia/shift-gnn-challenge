# Configuration Google Form pour Soumissions Privées

Ce guide explique comment configurer le système de soumissions via Google Form pour garder les soumissions privées tout en affichant uniquement les scores/rangs sur le leaderboard public.

---

## 📋 Prérequis

1. **Google Form créé** avec les champs suivants :
   - **Team Name** (Nom d'équipe) - texte court
   - **Email** - validation email
   - **Model Type** - liste déroulante : `human`, `llm`, `human+llm`
   - **CSV File** (Fichier CSV) - upload de fichier
   - **Notes** (optionnel) - texte long

2. **Google Form configuré** :
   - ✅ Limiter à **1 réponse par personne** (connexion Google requise)
   - ✅ Collecter les emails des répondants
   - ✅ Réponses stockées dans **Google Sheets**

---

## 🔧 Étape 1 : Configurer Google Cloud API

### 1.1 Créer un projet Google Cloud

1. Aller sur [Google Cloud Console](https://console.cloud.google.com/)
2. Créer un nouveau projet (ou utiliser un existant)
3. Noter le **Project ID**

### 1.2 Activer les APIs nécessaires

1. Dans le menu, aller à **APIs & Services** → **Library**
2. Activer :
   - **Google Sheets API**
   - **Google Drive API**

### 1.3 Créer un compte de service

1. Aller à **APIs & Services** → **Credentials**
2. Cliquer sur **Create Credentials** → **Service Account**
3. Nommer le compte (ex: `leaderboard-processor`)
4. Cliquer sur **Create and Continue**
5. Rôle : **Editor** (ou plus restrictif si possible)
6. Cliquer sur **Done**

### 1.4 Générer une clé JSON

1. Dans la liste des comptes de service, cliquer sur celui créé
2. Aller à l'onglet **Keys**
3. Cliquer sur **Add Key** → **Create new key**
4. Choisir **JSON**
5. Télécharger le fichier JSON (garder-le secret !)

### 1.5 Partager le Google Sheet avec le compte de service

1. Ouvrir le Google Sheet lié à ton formulaire
2. Cliquer sur **Share** (Partager)
3. Ajouter l'**email du compte de service** (trouvable dans le JSON téléchargé, champ `client_email`)
4. Donner les permissions **Viewer** (lecture seule)

### 1.6 Partager le dossier Google Drive (si nécessaire)

Si les fichiers CSV sont dans un dossier Drive :
1. Ouvrir le dossier dans Google Drive
2. Partager avec l'email du compte de service
3. Permissions : **Viewer**

---

## 🔧 Étape 2 : Obtenir l'ID du Google Sheet

1. Ouvrir le Google Sheet lié à ton formulaire
2. L'URL ressemble à : `https://docs.google.com/spreadsheets/d/SHEET_ID/edit`
3. Copier le **SHEET_ID** (la partie entre `/d/` et `/edit`)

Exemple :
```
https://docs.google.com/spreadsheets/d/1a2b3c4d5e6f7g8h9i0j/edit
                                    ↑ SHEET_ID = 1a2b3c4d5e6f7g8h9i0j
```

---

## 🔧 Étape 3 : Installer les dépendances

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

---

## 🔧 Étape 4 : Configurer les variables d'environnement

### Option A : Variables d'environnement (recommandé pour CI)

```bash
export GOOGLE_SHEETS_ID="ton_sheet_id_ici"
export GOOGLE_CREDENTIALS_PATH="/chemin/vers/service-account-key.json"
# OU
export GOOGLE_CREDENTIALS_JSON='{"type": "service_account", ...}'  # Contenu JSON complet
```

### Option B : Arguments en ligne de commande

```bash
python scripts/process_google_form_submissions.py \
    --sheets-id "ton_sheet_id_ici" \
    --credentials "/chemin/vers/service-account-key.json"
```

---

## 🔧 Étape 5 : Tester le script localement

```bash
# Tester la récupération des soumissions (sans push)
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID" \
    --credentials "path/to/service-account-key.json"

# Avec push automatique au repo
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID" \
    --credentials "path/to/service-account-key.json" \
    --push
```

---

## 🔧 Étape 6 : Automatiser avec GitHub Actions (optionnel)

Si tu veux automatiser le traitement, crée un workflow dans un **dépôt privé** :

**`.github/workflows/process_google_form.yml`** (dans dépôt privé) :

```yaml
name: Process Google Form Submissions

on:
  schedule:
    - cron: '*/30 * * * *'  # Toutes les 30 minutes
  workflow_dispatch:  # Déclenchement manuel

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout public repo
        uses: actions/checkout@v4
        with:
          repository: samuelmatia/shift-gnn-challenge  # Ton repo public
          token: ${{ secrets.PUBLIC_REPO_TOKEN }}
          path: public-repo
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
          pip install pandas pyarrow scikit-learn
      
      - name: Process submissions
        env:
          GOOGLE_SHEETS_ID: ${{ secrets.GOOGLE_SHEETS_ID }}
          GOOGLE_CREDENTIALS_JSON: ${{ secrets.GOOGLE_CREDENTIALS_JSON }}
        working-directory: public-repo
        run: |
          python scripts/process_google_form_submissions.py \
            --sheets-id "$GOOGLE_SHEETS_ID" \
            --push
      
      - name: Push leaderboard
        working-directory: public-repo
        run: |
          git config user.name "Leaderboard Bot"
          git config user.email "bot@example.com"
          git push origin main
```

**Secrets GitHub à configurer** (dans le dépôt privé) :
- `GOOGLE_SHEETS_ID` : ID de ton Google Sheet
- `GOOGLE_CREDENTIALS_JSON` : Contenu complet du fichier JSON du compte de service
- `PUBLIC_REPO_TOKEN` : Token GitHub avec permissions `contents:write` pour le repo public

---

## 🔧 Étape 7 : Mettre à jour le README

Modifier le README pour expliquer le nouveau processus de soumission via Google Form au lieu des PRs.

---

## 📝 Notes importantes

1. **Sécurité** : Ne jamais commiter le fichier JSON du compte de service dans le dépôt public
2. **Une seule soumission** : Le script vérifie les timestamps pour éviter les doublons
3. **Leaderboard uniquement** : Seuls `leaderboard.json` et `leaderboard.html` sont pushés au repo public
4. **Fichiers CSV** : Les fichiers CSV téléchargés restent dans le dépôt privé (ou local) et ne sont jamais pushés

---

## 🐛 Dépannage

### Erreur : "Google API libraries not installed"
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Erreur : "Permission denied" sur Google Sheet
- Vérifier que le compte de service a accès au Sheet (partagé avec son email)

### Erreur : "File not found" sur Google Drive
- Vérifier que le fichier CSV est accessible (partagé avec le compte de service si dans un dossier)

### Le script ne trouve pas de nouvelles soumissions
- Vérifier le nom de la colonne "Timestamp" dans Google Sheets (peut varier selon la langue)
