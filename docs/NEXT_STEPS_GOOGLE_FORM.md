# 🚀 Prochaines Étapes - Google Form Setup

Tu as créé ton Google Form ? Voici les étapes pour l'intégrer :

---

## ✅ Checklist rapide

- [ ] **Étape 1** : Configurer Google Cloud API (5-10 min)
- [ ] **Étape 2** : Obtenir l'ID de ton Google Sheet (1 min)
- [ ] **Étape 3** : Installer les dépendances Python (1 min)
- [ ] **Étape 4** : Tester le script localement (2 min)
- [ ] **Étape 5** : (Optionnel) Automatiser avec GitHub Actions

---

## 📝 Étape 1 : Configurer Google Cloud API

### 1.1 Créer un projet et activer les APIs

1. Va sur [Google Cloud Console](https://console.cloud.google.com/)
2. Créer un nouveau projet → nomme-le (ex: "shift-gnn-leaderboard")
3. Dans le menu → **APIs & Services** → **Library**
4. Recherche et active :
   - ✅ **Google Sheets API**
   - ✅ **Google Drive API**

### 1.2 Créer un compte de service

1. **APIs & Services** → **Credentials**
2. **Create Credentials** → **Service Account**
3. Nom : `leaderboard-processor`
4. Rôle : **Editor** (ou plus restrictif)
5. **Done**

### 1.3 Télécharger la clé JSON

1. Clique sur le compte de service créé
2. Onglet **Keys** → **Add Key** → **Create new key**
3. Format : **JSON**
4. **Télécharge le fichier** (garder-le secret !)

### 1.4 Partager le Google Sheet avec le compte de service

1. Ouvre ton **Google Sheet** (lié au formulaire)
2. **Share** (Partager)
3. Ajoute l'**email du compte de service** (trouvable dans le JSON téléchargé, champ `client_email`)
4. Permissions : **Viewer** (lecture seule)

---

## 📝 Étape 2 : Obtenir l'ID du Google Sheet

1. Ouvre ton Google Sheet
2. L'URL ressemble à :
   ```
   https://docs.google.com/spreadsheets/d/1a2b3c4d5e6f7g8h9i0j/edit
   ```
3. Copie la partie entre `/d/` et `/edit` → **C'est ton SHEET_ID**

---

## 📝 Étape 3 : Installer les dépendances

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/gnn-role-transition-challenge"
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

---

## 📝 Étape 4 : Tester le script

### Test sans push (recommandé pour la première fois)

```bash
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID_ICI" \
    --credentials "/chemin/vers/service-account-key.json"
```

**Remplace :**
- `TON_SHEET_ID_ICI` : l'ID de ton Google Sheet (étape 2)
- `/chemin/vers/service-account-key.json` : le chemin vers le fichier JSON téléchargé (étape 1.3)

### Si ça fonctionne, tester avec push

```bash
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID_ICI" \
    --credentials "/chemin/vers/service-account-key.json" \
    --push
```

---

## 📝 Étape 5 : Vérifier les colonnes du formulaire

Le script cherche ces noms de colonnes dans Google Sheets (peuvent varier selon la langue) :

- **Timestamp** : `Timestamp` ou `timestamp`
- **Team Name** : `Team Name`, `team_name`, `Nom d'équipe`
- **Email** : `Email` ou `email`
- **Model Type** : `Model Type` ou `model_type`
- **CSV File** : `CSV File`, `csv_file`, `Fichier CSV`

**Si tes colonnes ont d'autres noms**, modifie le script `process_google_form_submissions.py` aux lignes 80-85.

---

## 🔄 Automatisation (Optionnel)

Si tu veux que le script s'exécute automatiquement toutes les 30 minutes, crée un workflow GitHub Actions dans un **dépôt privé** (voir `docs/GOOGLE_FORM_SETUP.md` section "Étape 6").

---

## ❓ Problèmes courants

### "Google API libraries not installed"
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### "Permission denied" sur le Sheet
→ Vérifie que le compte de service a accès au Sheet (étape 1.4)

### "Could not extract file ID"
→ Vérifie que le lien du fichier CSV dans Google Sheets est un lien Google Drive valide

---

## 📚 Documentation complète

Pour plus de détails, voir : `docs/GOOGLE_FORM_SETUP.md`
