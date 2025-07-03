import json

import torch
import numpy as np

from safetensors.torch import load_file
from transformers import PreTrainedTokenizer


class YourTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", eos_token="</s>"):
        # Define vocabulary mapping amino acids & special tokens
        self.vocab = {
            "A": 5, "L": 6, "G": 7, "V": 8,
            # ... rest of your vocabulary
            pad_token: 0, eos_token: 1, unk_token: 2
        }

        # Reverse vocabulary for decoding
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

        # Set special tokens explicitly
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.eos_token = eos_token

        # Preprocessing
        self.characters_to_replace = "UZOB"
        self.replacement_character = "X"
        self.uses_whitespaces = False

        # Initialize parent class properly
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            eos_token=eos_token,
            vocab=vocab
        )


""" Rest of your tokenizer functionality... """


def to_biotrainer_config(self):
    json_output = {
        "vocab": self.vocab,
        "unk_token": self.unk_token,
        "pad_token": self.pad_token,
        "eos_token": self.eos_token,
        "characters_to_replace": self.characters_to_replace,
        "replacement_character": self.replacement_character,
        "uses_whitespaces": self.uses_whitespaces,
    }
    with open("custom_tokenizer_config.json", "w") as f:
        json.dump(json_output, f)


class YourModel(torch.nn.Module):
    def __init__(
            self,
            num_layers=6,
            hidden_size=512,
            num_heads=8,
    ):
        super(YourModel, self).__init__()


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        """ Adapt your model to accept exactly these to input parameters """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.hidden_states


def compare_outputs(original, onnx, rtol=1e-3, atol=1e-3):
    """Compare outputs from original and ONNX model"""
    np.testing.assert_allclose(original, onnx, rtol=rtol, atol=atol)
    print("✓ Outputs are identical within numerical precision")


def test_sequences():
    return [
        "MALLHSARVLSGVASAFHPGLAAAASARASSWWAHVEMGPPDPILGVTEAYKRDTNSKKMNLGVGAYRDDNGKPYVLPSVRKAEAQIAAKGLDKEYLPIGGLAEFCRASAELALGENSEVVKSGRFVTVQTISGTGALRIGASFLQRFFKFSRDVFLPKPSWGNHTPIFRDAGMQLQSYRYYDPKTCGFDFTGALEDISKIPEQSVLLLHACAHNPTGVDPRPEQWKEIATVVKKRNLFAFFDMAYQGFASGDGDKDAWAVRHFIEQGINVCLCQSYAKNMGLYGERVGAFTVICKDADEAKRVESQLKILIRPMYSNPPIHGARIASTILTSPDLRKQWLQEVKGMADRIIGMRTQLVSNLKKEGSTHSWQHITDQIGMFCFTGLKPEQVERLTKEFSIYMTKDGRISVAGVTSGNVGYLAHAIHQVTK",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
        "MSYYHHHHHHLESTSLYKKAGSENLYFQGSMINIERLKNLVVVKRKGILADGNDMPRVYCVNDDIVLGLMYSGKVKSFLEFMRTHGISRPAVTLKNNHEYEVISTYVRSFLGESAPVLAAYRRGKRRPIQVVYLSSPQGLQYPLGRMLKDAIQNYVRTIPIGGTGKFYTAGRPVNITVNCDMTGQMCNVSEEDVRNAGQPFHHIPEGIKVGQVGLKFYGNMTREEAEKIIKKLLPRHMRIGTIGHVDHGKTTLTAAITKVLADKGCGKQTREHKFLPGCSAGQNVGLLLRGIGKKDIMERGIKVGDIEIIVGLKEETPTLCQGNVSVKPGTKFTAQIEIYLTKQGVYEKKDVNCIRVPDCPPPTRAEIEEIIKRAELLLGKPVLYGSALQGLDRLVVVKNKTDLVEPQTIEKAIEKFDLYANVYEIIDQLPFNQAFKLLLELPDEFVIEKYRKELYQGVEIGEPIPVTPSQLTINKLIGHLEKANLSSRRQVAVKESIVYINELISSCEEAGWTKDLLGSIIEMTDKSLRVKLVKPEDKKQRLVFNLLDKIPKKEIYVHTLHGLGIELVDTPGHESFSNLRNRGRSITIELNDPHNKDSQGGSGEDNNVLSIKKYVQMLKKYNLYRM"
    ]


def prepare_input(sequence: str, tokenizer: YourTokenizer):
    """Prepare input for both models"""
    encoded = tokenizer.batch_encode_plus(
        [sequence],
        add_special_tokens=True,
        padding="longest",
        return_tensors="pt"
    )
    return encoded['input_ids'], encoded['attention_mask']


def convert_to_onnx(model, tokenizer, onnx_path="model.onnx"):
    """Convert PyTorch model to ONNX format"""
    model.eval()

    # Wrap the model
    wrapped_model = OnnxWrapper(model)
    wrapped_model.eval()

    # Create dummy input
    dummy_input_ids, dummy_attention_mask = prepare_input("ACDEF", tokenizer)

    # Export the model
    torch.onnx.export(
        wrapped_model,  # wrapped model being run
        (dummy_input_ids, dummy_attention_mask),  # model inputs as tuple
        onnx_path,  # where to save the model
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size', 1: 'sequence_length', 2: 'hidden_size'}
        }
    )

    print(f"Model exported to {onnx_path}")


def test_models(pytorch_model, tokenizer, onnx_path="model.onnx"):
    """Test and compare PyTorch and ONNX model outputs"""
    import onnxruntime

    # Create ONNX Runtime session
    ort_session = onnxruntime.InferenceSession(onnx_path)

    # Test sequences
    sequences = test_sequences()

    for idx, seq in enumerate(sequences):
        print(f"\nTesting sequence {idx + 1}/{len(sequences)} (length: {len(seq)})")

        # Prepare input
        input_ids, attention_mask = prepare_input(seq, tokenizer)

        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).hidden_states.numpy()

        # ONNX Runtime inference
        ort_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy()
        }
        print("ONNX model inputs:", [i.name for i in ort_session.get_inputs()])
        ort_output = ort_session.run(None, ort_inputs)[0]

        # Compare outputs
        try:
            compare_outputs(pytorch_output, ort_output)
        except AssertionError as e:
            print(f"❌ Outputs differ: {e}")
            print(f"Max difference: {np.max(np.abs(pytorch_output - ort_output))}")


def main():
    print("Loading PyTorch model...")
    tokenizer = YourTokenizer()
    model = YourModel(hidden_size=600, num_layers=16, num_heads=12)
    state_dict = load_file(f"your_model_checkpoint.safetensors")
    model.load_state_dict(state_dict)
    model.eval()

    print("\nConverting to ONNX...")
    convert_to_onnx(model, tokenizer)

    print("\nTesting models...")
    test_models(model, tokenizer)


if __name__ == "__main__":
    main()
