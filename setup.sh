# python3 get-pip.py
# export PATH=$PATH:/usr/local/bin
source ~/venvs/cot/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt