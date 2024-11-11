pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece
pip install cmake==3.30.4 lm-eval==0.3.0 ftfy accelerate
 


cd $PWD/3rdparty/flashinfer/python
pip install -e .
echo "flashinfer installed"

cd $PWD/3rdparty/Infinigen/speedup
pip install -e infinigen
pip install -e flexgen
echo "Infinigen installed"

cd $PWD/3rdparty/MoA
git checkout 0.0.1
pip install -e .
echo "MoA installed"

cd $PWD
python setup.py develop
