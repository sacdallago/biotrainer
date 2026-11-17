from biotrainer.embedding import EmbeddingAPI

embd_api = EmbeddingAPI(embedder_name="/path_to_model.onnx",
                        custom_tokenizer_config=None)  # TODO wip

sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "MALLSLLALVLLALPAGSAGPVPTPGQLAQEVTRLLEQ",
]

embd_gen = embd_api.generate_embeddings(input_data=sequences,
                            reduce=True)

for embd in embd_gen:
    print(embd)
