#!/bin/bash
# Script de test pour vérifier la configuration Google Drive

echo "🧪 Test de Configuration Google Drive"
echo "===================================="
echo ""

# Vérifier que gdown est installé
echo "1. Vérification de gdown..."
if ! python3 -c "import gdown" 2>/dev/null; then
    echo "   ⚠️  gdown n'est pas installé"
    echo "   Installation: pip install gdown"
    exit 1
else
    echo "   ✅ gdown est installé"
fi

# Demander l'ID du fichier
echo ""
echo "2. Configuration de l'ID du fichier"
echo "   Entrez l'ID du fichier Google Drive:"
read -r FILE_ID

if [ -z "$FILE_ID" ]; then
    echo "   ❌ ID du fichier vide"
    exit 1
fi

export PRIVATE_DATA_METHOD=google_drive
export GOOGLE_DRIVE_FILE_ID="$FILE_ID"

echo ""
echo "3. Test du téléchargement..."
echo "   ID utilisé: $FILE_ID"
echo ""

# Créer un backup du fichier existant si présent
if [ -f "data/private/test.parquet" ]; then
    echo "   📦 Sauvegarde du fichier existant..."
    cp data/private/test.parquet data/private/test.parquet.backup
fi

# Tester le téléchargement
python3 scripts/download_private_data.py

if [ $? -eq 0 ] && [ -f "data/private/test.parquet" ]; then
    SIZE=$(stat -f%z "data/private/test.parquet" 2>/dev/null || stat -c%s "data/private/test.parquet" 2>/dev/null)
    SIZE_MB=$(echo "scale=2; $SIZE / 1024 / 1024" | bc)
    echo ""
    echo "   ✅ Téléchargement réussi!"
    echo "   📊 Taille du fichier: ${SIZE_MB} MB"
    echo ""
    echo "4. Vérification du contenu..."
    python3 -c "import pandas as pd; df = pd.read_parquet('data/private/test.parquet'); print(f'   ✅ Fichier valide: {len(df)} lignes')" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Configuration réussie! Vous pouvez maintenant configurer GitHub Secrets."
        echo ""
        echo "Secrets à configurer sur GitHub:"
        echo "  - PRIVATE_DATA_METHOD = google_drive"
        echo "  - GOOGLE_DRIVE_FILE_ID = $FILE_ID"
    else
        echo "   ⚠️  Le fichier téléchargé semble invalide"
    fi
else
    echo ""
    echo "   ❌ Échec du téléchargement"
    echo "   Vérifiez:"
    echo "   - Que l'ID du fichier est correct"
    echo "   - Que le fichier est partagé avec 'Toute personne avec le lien'"
    echo "   - Votre connexion internet"
    
    # Restaurer le backup
    if [ -f "data/private/test.parquet.backup" ]; then
        mv data/private/test.parquet.backup data/private/test.parquet
        echo "   📦 Fichier original restauré"
    fi
    exit 1
fi

