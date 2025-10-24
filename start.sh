#!/bin/bash
echo "🚀 Lancement de l'application Flask..."

# Active ton environnement virtuel si présent
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Installe les dépendances
pip install -r requirements.txt

# Lancer Flask
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
