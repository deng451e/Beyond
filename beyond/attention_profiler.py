import torch 

class attention_profiler_:
    def __init__(
        self,
        proifling_layers_idx,
        model
        
    ):
      
        self.model = model
        self.proifling_layers_idx  = proifling_layers_idx
      

    def __call__(self, past_key_values,query):