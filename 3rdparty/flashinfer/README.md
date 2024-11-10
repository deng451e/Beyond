 
### Installation

We provide prebuilt wheels for Linux and you can try out FlashInfer with the following command:

```bash
build from source:

```bash
git clone git@github.com:deng451e/flashinfer.git --recursive
cd flashinfer/python
pip install -e .
```

to reduce binary size during build and testing:
```bash
git clone git@github.com:deng451e/flashinfer.git --recursive
cd flashinfer/python
# ref https://pytorch.org/docs/stable/generated/torch.cuda.get_device_capability.html#torch.cuda.get_device_capability
export TORCH_CUDA_ARCH_LIST=8.0
pip install -e .
```
 