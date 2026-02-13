# ✅ Prochaines Étapes Finales - Google Form

Tes colonnes sont configurées ! Voici les étapes pour finaliser :

---

## 📋 Checklist

- [x] Google Form créé avec colonnes : Horodateur, Adresse e-mail, 1. Team Name, 2. Model Type, 3. Submission File ( .csv)
- [ ] **Étape 1** : Configurer Google Cloud API (5-10 min)
- [ ] **Étape 2** : Obtenir l'ID du Google Sheet (1 min)
- [ ] **Étape 3** : Installer les dépendances (1 min)
- [ ] **Étape 4** : Tester le script (2 min)
- [ ] **Étape 5** : Mettre à jour le README avec le lien du formulaire

---

## 🔧 Étape 1 : Configurer Google Cloud API

### 1.1 Créer un projet Google Cloud

1. Va sur [Google Cloud Console](https://console.cloud.google.com/)
2. Crée un nouveau projet (ex: "shift-gnn-leaderboard")
3. Note le **Project ID**

### 1.2 Activer les APIs

1. Menu → **APIs & Services** → **Library**
2. Active :
   - ✅ **Google Sheets API**
   - ✅ **Google Drive API**

### 1.3 Créer un compte de service

1. **APIs & Services** → **Credentials**
2. **Create Credentials** → **Service Account**
3. Nom : `leaderboard-processor`
4. Rôle : **Editor**
5. **Done**

### 1.4 Télécharger la clé JSON

1. Clique sur le compte de service créé
2. **Keys** → **Add Key** → **Create new key**
3. Format : **JSON**
4. **Télécharge le fichier** (garder secret !)

### 1.5 Partager le Google Sheet

1. Ouvre ton **Google Sheet** (lié au formulaire)
2. **Partager** (Share)
3. Ajoute l'**email du compte de service** (dans le JSON téléchargé, champ `client_email`)
4. Permissions : **Lecteur** (Viewer)

---

## 🔧 Étape 2 : Obtenir l'ID du Google Sheet

1. Ouvre ton Google Sheet
2. L'URL ressemble à :
   ```
   https://docs.google.com/spreadsheets/d/1a2b3c4d5e6f7g8h9i0j/edit
   ```
3. Copie la partie entre `/d/` et `/edit` → **C'est ton SHEET_ID**

---

## 🔧 Étape 3 : Installer les dépendances

```bash
cd "/home/sam/Desktop/GNNs BASIRA Lab/gnn-role-transition-challenge"
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

---

## 🔧 Étape 4 : Tester le script

### Test sans push (première fois)

```bash
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID" \
    --credentials "/chemin/vers/service-account-key.json"
```

**Remplace :**
- `TON_SHEET_ID` : l'ID de ton Google Sheet (étape 2)
- `/chemin/vers/service-account-key.json` : chemin vers le fichier JSON téléchargé (étape 1.4)

### Ce que le script va faire :

1. ✅ Lire les réponses depuis Google Sheets
2. ✅ Afficher les colonnes disponibles (pour vérification)
3. ✅ Télécharger les fichiers CSV depuis Google Drive
4. ✅ Évaluer chaque soumission
5. ✅ Mettre à jour `leaderboard.json` et `leaderboard.html`

### Si ça fonctionne, tester avec push :

```bash
python scripts/process_google_form_submissions.py \
    --sheets-id "TON_SHEET_ID" \
    --credentials "/chemin/vers/service-account-key.json" \
    --push
```

---

## 🔧 Étape 5 : Mettre à jour le README

Modifie le README pour remplacer les instructions de PR par un lien vers ton Google Form.

---

## 📝 Notes importantes

1. **Colonnes configurées** : Le script utilise maintenant tes noms exacts :
   - `Horodateur` (Timestamp)
   - `Adresse e-mail` (Email)
   - `1. Team Name`
   - `2. Model Type`
   - `3. Submission File ( .csv)`

2. **Fichiers CSV** : Google Forms stocke les fichiers dans Google Drive. Le script télécharge automatiquement depuis Drive.

3. **Une seule soumission** : Le script vérifie les timestamps pour éviter les doublons.

4. **Leaderboard uniquement** : Seuls `leaderboard.json` et `leaderboard.html` sont pushés au repo public (pas les CSV).

---

## 🐛 Dépannage

### Le script affiche "Available columns" mais ne trouve pas les données
→ Vérifie que les noms de colonnes correspondent exactement (y compris les espaces et la casse)

### "Permission denied" sur Google Sheet
→ Vérifie que le compte de service a accès au Sheet (étape 1.5)

### "Could not extract file ID"
→ Vérifie que la colonne "3. Submission File ( .csv)" contient bien un lien Google Drive

---

## 🎯 Une fois que ça fonctionne

Tu peux :
- Exécuter le script manuellement quand tu veux traiter les nouvelles soumissions
- Ou automatiser avec GitHub Actions (voir `docs/GOOGLE_FORM_SETUP.md`)
