# Configuration File Options Overview

Here, you can find an overview about all configuration options available in *biotrainer*.
For more details, please refer to the descriptions in [config_file_options](config_file_options.md).

```yaml
# General Options
protocol: residue_to_class | residue_to_value | residues_to_class | residues_to_value | sequence_to_class | sequence_to_value
interaction: multiply | concat  # Default: None
seed: 1234  # Default: 42
device: cpu | cuda | cuda:0 | cuda:1  # Default: Uses cuda if available, otherwise cpu
save_split_ids: True | False  # Default: False
ignore_file_inconsistencies: True | False  # Default: False
output_dir: path/to/output/directory  # Default: output
bootstrapping_iterations: 55  # Default: 30, Disable: 0, note that sanity checks will always use bootstrapping
sanity_check: True | False  # Default: True
external_writer: tensorboard | none  # Default: tensorboard, none deactivates it

# Input File
input_file: path/to/input_file.fasta  # Required for all protocols (unless huggingface dataset is used)

# Embeddings
embedder_name: Rostlab/prot_t5_xl_uniref50 | ElnaggarLab/ankh-large | user/your-hf-model | your_model.onnx | one_hot_encoding | random_embedder | AAOntology | blosum62
use_half_precision: True | False  # Default: False
embeddings_file: path/to/embeddings.h5  # Optional pre-computed embeddings file
dimension_reduction_method: umap | tsne  # Default: None, only possible for per-sequence embeddings
n_reduced_components: 5  # Default: None, requires dimension_reduction_method to be set
custom_tokenizer_config: tokenizer_config.json  # If no config is provided, the default T5Tokenizer is used. Only applicable if using an onnx embedder

# Model Parameters
model_choice: FNN | CNN | LogReg | LightAttention  # Protocol-dependent default
optimizer_choice: adam  # Default: adam
learning_rate: 1e-3  # Default: 1e-3
dropout_rate: 0.25  # Default: 0.25
loss_choice: cross_entropy_loss | mean_squared_error  # Protocol-dependent default
use_class_weights: True | False  # Default: False
disable_pytorch_compile: True | False  # Default: True

# Training Parameters
num_epochs: 200  # Default: 200
patience: 10  # Default: 10
epsilon: 1e-3  # Default: 1e-3
batch_size: 128  # Default: 128
shuffle: True | False  # Default: True

# Cross Validation
cross_validation_config:
  method: hold_out | k_fold | leave_p_out
  
  # k-fold specific options
  k: 5  # Required for k-fold, k >= 2
  stratified: True | False  # Default: False
  repeat: 3  # Default: 1
  nested: True | False  # Default: False
  nested_k: 3  # Required for nested k-fold, nested_k >= 2
  search_method: random_search | grid_search
  n_max_evaluations_random: 3  # For random search
  
  # leave-p-out specific option
  p: 5  # p >= 1
  
  # Common option
  choose_by: loss | accuracy | precision | recall  # Default: loss

# Special Training Modes
auto_resume: True | False  # Default: False
pretrained_model: path/to/model_checkpoint.safetensors  # Mutually exclusive with auto_resume
limited_sample_size: 100  # Default: -1 (all options)

# Finetuning
finetuning_config:
  method: lora  # Only lora supported at the moment
  random_masking: True | False  # Default: False, if True, random_masking will be applied for masked language modeling.
  lora_r: 8  # Lora Rank
  lora_alpha: 16  # Lora Alpha
  lora_dropout: 0.05  # Lora dropout probability
  lora_target_modules: auto | ["query", "key", "value"]  # Names of target modules for lora (ESM), ProtT5: ["q", "k", "v", "o"], Auto: Infer automatically from model name
  lora_bias: none | all | lora_only  # Bias type for lora

# HuggingFace Dataset Integration
hf_dataset:
  path: huggingface_user_name/repository_name  # Required
  subset: subset_name  # Optional
  sequence_column: sequences_column_name  # Required
  target_column: targets_column_name  # Required
  mask_column: mask_column_name  # Optional
```