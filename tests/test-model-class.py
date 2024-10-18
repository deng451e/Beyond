import torch
from transformers import BertModel,BertConfig,LlamaConfig,LlamaModel
import types
 
# Load a pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')

# List to store layer indices during forward pass
layer_execution_indices = []

config  = BertConfig()
 

def increase(self,):
    with torch.cuda.stream(self.cpu_stream):
        print( self.attn_layer_idx)
        self.attn_layer_idx  = (self.attn_layer_idx+1)%self.attn_layer_num
        
 

cpu_stream =  torch.cuda.Stream()
model.cpu_stream = cpu_stream
model.attn_layer_idx = 0
model.attn_layer_num = config.num_hidden_layers
model.increase =  types.MethodType(increase,model)

# Hook function to capture layer index
def hook_fn(module, input, output, layer_idx,instance):
    
    instance.increase()
     
# Register the hook for each encoder layer
for idx, layer in enumerate(model.encoder.layer):
    # Add the layer index to the hook function arguments
    layer.register_forward_hook(lambda module, input, output, idx=idx: hook_fn(module, input, output, idx,model))

# Dummy input (for demonstration purposes)
input_ids = torch.tensor([[101, 2054, 2003, 1996, 102]])  # Tokenized input for: "[CLS] What is the [SEP]"

# Perform a forward pass, hooks will be triggered
outputs = model(input_ids)