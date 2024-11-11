import torch 
import torch  
import flashinfer 
import argparse
import time 
from beyond.utils import *
from beyond.loading import *
 
class attention_profiler_:
    def __init__(
        self,
        num_layers=0,
        alpha=1e+1,
        cpu_stream=None,
        module=None
        
    ):
      
        self.num_layers = num_layers
        self.alpha = alpha
        self.proifling_layers   = [dict() for _ in range(num_layers)]
        self.module = module

    def __call__(self,attention_weights):
            layer_idx = self.module.attn_layer_idx
            indices = (attention_weights> self.alpha).nonzero(as_tuple=True)
            for indice in indices:
                if indice not in self.proifling_layers[layer_idx]:
                    self.proifling_layers[layer_idx][indice] = 0
                if attention_weights> self.alpha:
                     self.proifling_layers[layer_idx][indice] += 1 

    def get_attn_weights(self,idx):
        return list(self.proifling_layers[idx].values())







def add_profiler_hook(model,profiler,attention_method):
    global layer_idx
    config = model.config 
     
    layer_idx = config.num_hidden_layers-1
   
    profiler.num_layers=config.num_hidden_layers
    profiler.copy_stream= torch.cuda.Stream()
   
      
    def add_hook(model):
        for name, module in reversed(model._modules.items()):
            if len(list(module.children())) > 0:
                add_hook( module,)

             
            if isinstance(module, attention_method):
                global layer_idx
                model._modules[name].attn_layer_idx = layer_idx
                model._modules[name].register_forward_hook(profiler)
                layer_idx -= 1 
    add_hook(model)
 
 




 
    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
   
    args = parser.parse_args()
    attention_method = None
     
    match args.model.split('/')[-1].split('-')[0]:
        case "llama": 
            from transformers.models.llama.modeling_llama import LlamaAttention
            attention_method = LlamaAttention
            
        case "opt":
            from transformers.models.opt.modeling_opt import OPTAttention
            attention_method = OPTAttention
             
    profiler=attention_profiler_()
    model, tokenizer = load_model(args.model)
    add_profiler_hook(model,profiler,attention_method)
    