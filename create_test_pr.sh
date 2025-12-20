#!/bin/bash
# Script pour créer rapidement une Pull Request de test

echo "🧪 Création d'une Pull Request de Test pour le Leaderboard"
echo "=========================================================="
echo ""

# Demander le nom de l'équipe
read -p "Entrez le nom de votre équipe (ex: awesome_team): " TEAM_NAME

if [ -z "$TEAM_NAME" ]; then
    echo "❌ Le nom de l'équipe ne peut pas être vide"
    exit 1
fi

# Nom du fichier de soumission
SUBMISSION_FILE="submissions/${TEAM_NAME}.csv"

# Vérifier si le fichier existe déjà
if [ -f "$SUBMISSION_FILE" ]; then
    echo "⚠️  Le fichier $SUBMISSION_FILE existe déjà"
    read -p "Voulez-vous le remplacer? (y/n): " REPLACE
    if [ "$REPLACE" != "y" ]; then
        echo "❌ Annulé"
        exit 1
    fi
fi

# Créer le fichier de soumission à partir d'un exemple
if [ -f "submissions/sample_submission_1.csv" ]; then
    cp submissions/sample_submission_1.csv "$SUBMISSION_FILE"
    echo "✅ Fichier de soumission créé: $SUBMISSION_FILE"
else
    echo "❌ Fichier exemple non trouvé: submissions/sample_submission_1.csv"
    exit 1
fi

# Créer une branche
BRANCH_NAME="test-submission-${TEAM_NAME}"
echo ""
echo "📦 Création de la branche: $BRANCH_NAME"

# Vérifier si on est déjà sur une branche de test
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" == "main" ]] || [[ "$CURRENT_BRANCH" == "master" ]]; then
    git checkout -b "$BRANCH_NAME"
else
    read -p "Vous êtes sur la branche '$CURRENT_BRANCH'. Créer quand même une nouvelle branche? (y/n): " CREATE_NEW
    if [ "$CREATE_NEW" == "y" ]; then
        git checkout -b "$BRANCH_NAME"
    else
        BRANCH_NAME="$CURRENT_BRANCH"
        echo "Utilisation de la branche actuelle: $BRANCH_NAME"
    fi
fi

# Ajouter et commiter
echo ""
echo "📝 Ajout du fichier..."
git add "$SUBMISSION_FILE"

echo "💾 Commit..."
git commit -m "Add test submission: $TEAM_NAME"

# Demander si on veut pousser
echo ""
read -p "Voulez-vous pousser la branche sur GitHub maintenant? (y/n): " PUSH_NOW

if [ "$PUSH_NOW" == "y" ]; then
    echo ""
    echo "🚀 Push de la branche..."
    git push origin "$BRANCH_NAME"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Branche poussée avec succès!"
        echo ""
        echo "📋 Prochaines étapes:"
        echo "1. Allez sur GitHub: https://github.com/samuelmatia/gnn-role-transition-challenge"
        echo "2. Vous verrez une bannière 'Compare & pull request' - cliquez dessus"
        echo "3. Remplissez le formulaire et créez la PR"
        echo "4. Le workflow GitHub Actions évaluera automatiquement votre soumission"
        echo ""
        echo "Ou utilisez ce lien direct (remplacez USERNAME si nécessaire):"
        echo "https://github.com/samuelmatia/gnn-role-transition-challenge/compare/main...$BRANCH_NAME"
    else
        echo ""
        echo "⚠️  Erreur lors du push. Vérifiez:"
        echo "   - Que vous êtes connecté à GitHub (git remote -v)"
        echo "   - Que vous avez les permissions"
        echo ""
        echo "Vous pouvez pousser manuellement avec:"
        echo "   git push origin $BRANCH_NAME"
    fi
else
    echo ""
    echo "📋 Pour pousser plus tard, exécutez:"
    echo "   git push origin $BRANCH_NAME"
fi

echo ""
echo "✨ Terminé! Bon test!"

