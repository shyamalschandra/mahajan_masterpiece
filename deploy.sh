#!/bin/bash

# GitHub Pages Deployment Script
# This script helps you deploy your Jekyll site to GitHub Pages

echo "🚀 GitHub Pages Deployment Script"
echo "=================================="
echo ""

# Check if remote is already configured
if git remote -v | grep -q "origin"; then
    echo "✅ Remote 'origin' is already configured:"
    git remote -v
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
    else
        echo "Using existing remote configuration."
        exit 0
    fi
fi

# Get repository information
echo "📋 Repository Setup"
echo "------------------"
echo ""
echo "Your repository should be: https://github.com/shyamalschandra/mahajan_masterpiece"
echo ""
read -p "Have you created the repository on GitHub? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "⚠️  Please create the repository first:"
    echo "   1. Go to: https://github.com/new"
    echo "   2. Repository name: mahajan_masterpiece"
    echo "   3. Owner: shyamalschandra"
    echo "   4. Visibility: Public"
    echo "   5. DO NOT initialize with README/gitignore/license"
    echo "   6. Click 'Create repository'"
    echo ""
    read -p "Press Enter when repository is created..."
fi

# Add remote
echo ""
echo "🔗 Adding remote repository..."
git remote add origin https://github.com/shyamalschandra/mahajan_masterpiece.git

if [ $? -eq 0 ]; then
    echo "✅ Remote added successfully"
    git remote -v
else
    echo "❌ Failed to add remote. It may already exist."
    echo "   Run: git remote -v to check"
    exit 1
fi

# Set branch to main
echo ""
echo "🌿 Setting branch to 'main'..."
git branch -M main

# Push to GitHub
echo ""
echo "📤 Pushing to GitHub..."
echo "   (You may be prompted for GitHub credentials)"
echo ""
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo ""
    echo "📝 Next Steps:"
    echo "   1. Go to: https://github.com/shyamalschandra/mahajan_masterpiece/settings/pages"
    echo "   2. Under 'Source', select:"
    echo "      - Branch: main"
    echo "      - Folder: / (root)"
    echo "   3. Click 'Save'"
    echo ""
    echo "🌐 Your site will be live at:"
    echo "   https://shyamalschandra.github.io/mahajan_masterpiece/"
    echo ""
    echo "⏳ First deployment takes 1-5 minutes"
    echo "   Check progress: https://github.com/shyamalschandra/mahajan_masterpiece/actions"
else
    echo ""
    echo "❌ Push failed. Common issues:"
    echo "   - Authentication: Use Personal Access Token (not password)"
    echo "   - Repository doesn't exist: Create it first on GitHub"
    echo "   - Network issues: Check your internet connection"
    echo ""
    echo "💡 To create Personal Access Token:"
    echo "   GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)"
    echo "   Required scope: repo (full control)"
fi

