# Mapping des Colonnes Google Form

## Colonnes de ton formulaire

D'après ton formulaire, les colonnes sont :

1. **Timestamp** (automatique par Google Forms)
2. **Team Name**
3. **Model Type** (valeurs : `human`, `llm`, `human+llm`)
4. **Submission File ( .csv)** (upload de fichier)

## Note importante sur l'Email

Google Forms collecte automatiquement l'email si tu as activé "Collect email addresses" dans les paramètres du formulaire. Cette colonne s'appelle généralement **"Email Address"** dans Google Sheets.

Si tu n'as pas activé la collecte d'emails, cette colonne n'existera pas et le script utilisera une valeur vide.

## Format du fichier CSV uploadé

Google Forms stocke les fichiers uploadés dans Google Drive. Dans Google Sheets, la colonne "Submission File ( .csv)" contient généralement :
- Un lien Google Drive (ex: `https://drive.google.com/file/d/FILE_ID/view`)
- Ou parfois juste le nom du fichier

Le script extrait automatiquement l'ID du fichier depuis le lien Drive pour le télécharger.

## Vérification

Pour vérifier les noms exacts des colonnes dans ton Google Sheet :

1. Ouvre ton Google Sheet
2. Regarde la première ligne (en-têtes)
3. Les noms doivent correspondre exactement à ceux utilisés dans le script

Si les noms diffèrent, modifie le script `process_google_form_submissions.py` aux lignes 209-213.
