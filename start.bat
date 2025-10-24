@echo off
echo 🚀 Lancement de l'application Flask...

REM Active ton environnement virtuel si tu en as un
if exist venv (
    call venv\Scripts\activate
)

REM Installe les dépendances
pip install -r requirements.txt

REM Démarre le serveur Flask
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
