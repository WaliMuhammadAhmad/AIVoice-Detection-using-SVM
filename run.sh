python3 -m venv env
source env/bin/activate
pip list
pip install -r requirements.txt
python3 train.py
python3 app.py