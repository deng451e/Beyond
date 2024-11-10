pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
pip install cmake==3.30.4
pip install -r requirements.txt

cd ./3rparty
git clone git@github.com:deng451e/flashinfer.git --recursive
cd flashinfer/python
pip install -e .
cd ../../../
python setup.py develop
