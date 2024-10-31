##  Attention Beyond the scope of GPU memory for LLMs inference 


 

### Usage

### Environment Setup

```bash
git clone --recurse-submodules git@github.com:deng451e/Beyond.git
cd Beyond 
conda create -yn beyond python=3.10
conda activate beyond
bash install.sh 
```

### Run Models 
  
```bash
cd run-models
# To run origianl models 
python run-llama.py 
python run-opt.py 
python run-gptNeox.py 
# To run origianl models  with Beyond 
python run-llama.py --enable_modify
python run-opt.py --enable_modify
python run-gptNeox.py --enable_modify
```