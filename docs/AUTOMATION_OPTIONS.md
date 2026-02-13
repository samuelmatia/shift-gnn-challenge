# 🤖 Options d'Automatisation du Leaderboard

## Option 1 : Google Apps Script + GitHub Actions Webhook (Recommandé)

Cette solution déclenche automatiquement le traitement dès qu'une nouvelle réponse arrive dans le Google Form.

### Comment ça marche

1. **Google Apps Script** dans ton Google Sheet détecte les nouvelles réponses
2. Appelle une **webhook GitHub Actions** (workflow_dispatch)
3. GitHub Actions exécute le script de traitement
4. Le leaderboard est mis à jour automatiquement

### Mise en place

#### Étape 1 : Créer un workflow GitHub Actions avec webhook

Crée `.github/workflows/process_google_form_webhook.yml` dans un **dépôt privé** :

```yaml
name: Process Google Form Submission (Webhook)

on:
  workflow_dispatch:
    inputs:
      timestamp:
        description: 'Timestamp of the new submission'
        required: true
        type: string

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
        working-directory: public-repo
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
            --sheet-name "Feuille1" \
            --push
```

#### Étape 2 : Créer un Personal Access Token GitHub

1. Va sur GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
2. **Generate new token** → Nomme-le (ex: "leaderboard-bot")
3. Permissions : `repo` (full control)
4. Copie le token (tu en auras besoin pour l'Apps Script)

#### Étape 3 : Créer le Google Apps Script

1. Ouvre ton **Google Sheet**
2. Menu → **Extensions** → **Apps Script**
3. Colle ce code :

```javascript
// Configuration
const GITHUB_TOKEN = 'ton_personal_access_token_github';
const GITHUB_REPO = 'samuelmatia/shift-gnn-challenge';  // Ton repo public
const GITHUB_WORKFLOW = 'process_google_form_webhook.yml';  // Nom du workflow
const GITHUB_OWNER = 'samuelmatia';  // Ton username GitHub

// Fonction appelée quand une nouvelle réponse arrive
function onFormSubmit(e) {
  const sheet = e.source.getActiveSheet();
  const lastRow = sheet.getLastRow();
  const timestamp = sheet.getRange(lastRow, 1).getValue(); // Colonne Horodateur
  
  Logger.log('New submission detected: ' + timestamp);
  
  // Appeler le webhook GitHub Actions
  triggerGitHubWorkflow(timestamp);
}

// Déclencher le workflow GitHub Actions
function triggerGitHubWorkflow(timestamp) {
  const url = `https://api.github.com/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/workflows/${GITHUB_WORKFLOW}/dispatches`;
  
  const payload = {
    ref: 'main',  // ou 'master' selon ta branche
    inputs: {
      timestamp: timestamp.toString()
    }
  };
  
  const options = {
    method: 'post',
    headers: {
      'Authorization': `token ${GITHUB_TOKEN}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    },
    payload: JSON.stringify(payload)
  };
  
  try {
    const response = UrlFetchApp.fetch(url, options);
    Logger.log('GitHub workflow triggered: ' + response.getResponseCode());
  } catch (error) {
    Logger.log('Error triggering workflow: ' + error.toString());
  }
}

// Installer le trigger (à exécuter une seule fois)
function installTrigger() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Feuille1');
  ScriptApp.newTrigger('onFormSubmit')
    .onFormSubmit()
    .create();
  Logger.log('Trigger installed successfully');
}
```

4. Remplace les valeurs dans le script :
   - `GITHUB_TOKEN` : ton Personal Access Token
   - `GITHUB_REPO` : ton repo (sans le .git)
   - `GITHUB_OWNER` : ton username GitHub
   - `GITHUB_WORKFLOW` : nom exact du fichier workflow

5. **Exécute `installTrigger()` une seule fois** :
   - Menu → **Run** → `installTrigger`
   - Autorise les permissions demandées

6. **Test** : Soumets une réponse de test au formulaire → le workflow GitHub devrait se déclencher automatiquement

---

## Option 2 : GitHub Actions avec Polling (Plus Simple)

Exécute le script périodiquement (toutes les 5-10 minutes) pour vérifier les nouvelles réponses.

### Avantages
- ✅ Plus simple à configurer
- ✅ Pas besoin de Google Apps Script
- ✅ Fonctionne même si le webhook échoue

### Inconvénients
- ⏱️ Latence de 5-10 minutes (pas instantané)

### Mise en place

Crée `.github/workflows/process_google_form_polling.yml` dans un **dépôt privé** :

```yaml
name: Process Google Form Submissions (Polling)

on:
  schedule:
    - cron: '*/5 * * * *'  # Toutes les 5 minutes
  workflow_dispatch:  # Déclenchement manuel

jobs:
  process:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout public repo
        uses: actions/checkout@v4
        with:
          repository: samuelmatia/shift-gnn-challenge
          token: ${{ secrets.PUBLIC_REPO_TOKEN }}
          path: public-repo
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        working-directory: public-repo
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
            --sheet-name "Feuille1" \
            --push
```

**Secrets GitHub à configurer** (dans le dépôt privé) :
- `GOOGLE_SHEETS_ID` : `1hSZlPR2GyXLbjbWurCZdBRRsAN_vB9LyiKq8XG6UTQI`
- `GOOGLE_CREDENTIALS_JSON` : Contenu complet du fichier JSON du compte de service
- `PUBLIC_REPO_TOKEN` : Personal Access Token GitHub avec permissions `repo`

---

## Comparaison

| Critère | Option 1 (Apps Script) | Option 2 (Polling) |
|---------|------------------------|---------------------|
| **Latence** | ⚡ Instantané | ⏱️ 5-10 minutes |
| **Complexité** | 🔴 Moyenne | 🟢 Simple |
| **Fiabilité** | 🟡 Dépend de Apps Script | 🟢 Très fiable |
| **Coût** | 🟢 Gratuit | 🟢 Gratuit (GitHub Actions) |

---

## Recommandation

**Pour commencer** : Utilise **Option 2 (Polling)** - plus simple et fiable.

**Pour une latence minimale** : Utilise **Option 1 (Apps Script)** une fois que tu es à l'aise avec la configuration.

---

## Notes importantes

1. **Dépôt privé requis** : Les workflows doivent être dans un dépôt privé pour garder les credentials secrets
2. **Limites GitHub Actions** : 2000 minutes/mois gratuites (suffisant pour polling toutes les 5 min)
3. **Sécurité** : Ne jamais commiter les credentials dans le repo public
