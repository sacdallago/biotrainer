# onnx_embedder example

These examples show how to convert a custom embedder model to onnx and use it within biotrainer.

* `onnx_conversion.py`: Example script that shows how to convert
* `custom_tokenizer_config.json`: Custom tokenizer config that can be read by biotrainer to tokenize protein sequences
* `config.yml` + `sequences.fasta` + `labels.fasta`: residue_to_class example that shows how to use the onnx config options 


Find more information about the ONNX format here:
* https://onnx.ai/
* https://pytorch.org/docs/stable/onnx.html
