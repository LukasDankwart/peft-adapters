import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import json

class RobertaAdapterModel:

    def __init__(self):
        self.model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        print(f"[DONE] Initializing RobertaAdapterModel.")


    def add_adapter_layers(self, adapter_name):
        if self.model is None:
            raise RuntimeError(f"Model state is none")
        self.model.add_adapter(adapter_name, config="houlsby")


    def add_classification_head(self, head_name, num_labels=2):
        if self.model is None:
            raise RuntimeError(f"Model state is none")
        self.model.add_classification_head(head_name, num_labels=num_labels)


    def get_model_state(self):
        if self.model is None :
            raise RuntimeError(f"Model state is none")
        return self.model


    def prepare_for_training(self, adapter_name):
        if self.model is None:
            raise RuntimeError("Model state is none")
        self.model.train_adapter(adapter_name)


if __name__ == "__main__":
    model = RobertaAdapterModel()
    adapter_name = "dummy_adapter"
    #adapter = AutoAdapterModel.from_pretrained("roberta-base")

    intern_model = model.get_model_state()
    model.add_adapter_layers(adapter_name=adapter_name)
    model.add_classification_head(head_name=adapter_name)
    model.prepare_for_training(adapter_name)
    intern_model.active_adapters = adapter_name

    first_layer = intern_model.roberta.encoder.layer[0]
    print(first_layer)

    attn_adapter_down = first_layer.attention.output.adapters[adapter_name].adapter_down[0]
    attn_adapter_up = first_layer.attention.output.adapters[adapter_name].adapter_up
    print(attn_adapter_down)
    print(attn_adapter_up)
    bottleneck_activation = first_layer
    out_adapter_down = first_layer.output.adapters[adapter_name].adapter_down[0]
    out_adapter_up = first_layer.output.adapters[adapter_name].adapter_up
    print(attn_adapter_down)
    print(attn_adapter_up)
    print(f"Number of layers: {len(intern_model.roberta.encoder.layer)}")
    attn_adapter_down.eval()
    attn_adapter_up.eval()
    n_iterations = 100
    sum_adapter_output = 0.0

    exit()

    # Dummy input for onnx export
    dummy_input_ids = torch.randint(0, 50265, (1, 16))
    dummy_attention_mask = torch.ones(1, 16)
    dummy_inputs = (dummy_input_ids, dummy_attention_mask)

    # Export ONNX of baseline (no adapters)
    baseline_model = model.model
    baseline_model.eval()
    baseline_model.cpu()

    print("Exporting baseline onnx...")
    torch.onnx.export(
        baseline_model,
        dummy_inputs,
        "roberta_baseline.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits']
    )
    print("Baseline onnx exported!\n")

    # Export ONNX of adapter variant
    adapter_model = RobertaAdapterModel()
    adapter_model.add_adapter_layers("dummy_adapter")
    adapter_pt_model = adapter_model.get_model_state()
    adapter_pt_model.active_adapters = "dummy_adapter"
    adapter_pt_model.eval()
    adapter_pt_model.cpu()
    print("Exporting adapter onnx...")
    torch.onnx.export(
        baseline_model,
        dummy_inputs,
        "roberta_adapter.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits']
    )
    print("Adapter model onnx exported!\n")

    print(f"Adapter-model params: {sum(p.numel() for p in adapter_pt_model.parameters())}")
    print(f"Baseline-model params: {sum(p.numel() for p in baseline_model.parameters())}")


    print("Active Adapter:", adapter_pt_model.active_adapters)
    print(adapter_pt_model.adapter_summary())
    adapter_config = adapter_pt_model.config.adapters.get("dummy_adapter")
    print(json.dumps(adapter_config, indent=4))
