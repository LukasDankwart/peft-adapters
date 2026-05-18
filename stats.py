from model.roberta import RobertaAdapterModel
from adapters import AdapterConfig

def show_parameter_stats():
    model = RobertaAdapterModel()
    base_parameters = sum(p.numel() for p in model.model.parameters())
    trainable_base_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\n ===== PARAMETER STATS =====")
    print(f"Base parameters: {base_parameters / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_base_parameters / 1e6:.2f}M \n")

    print(f"Adding adapter and set to train-adapter mode...")
    # Setting model to adapter training
    adapter_name = "adapter_0"
    model.add_adapter_layers(adapter_name)
    model.prepare_for_training(adapter_name)
    intern_model = model.get_model_state()
    intern_model.active_adapters = adapter_name
    adapter_base_parameters = sum(p.numel() for p in model.model.parameters())
    trainable_adapter_base_parameters = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    adapter_parameters = sum(p.numel() for (name, p) in intern_model.named_parameters() if adapter_name in name)
    print(f"Base + Adapter parameters: {adapter_base_parameters / 1e6:.2f}M")
    print(f"-- Base: {base_parameters / 1e6:.2f}M")
    print(f"-- Adapter: {adapter_parameters / 1e6:.2f}M")
    print(f"-- % growth: {adapter_base_parameters / base_parameters:.2f}%")
    print(f"Trainable parameters of Base + Adapter: {trainable_adapter_base_parameters / 1e6:.2f}M")
    print(f"-- % of all parameters: {adapter_parameters / adapter_base_parameters:.2f}% \n")
    print(f"There is a difference: TrainableParameters - AdapterParameters = {trainable_adapter_base_parameters - adapter_parameters}")
    print(f"Print trainable parameters: ")
    for name, param in intern_model.named_parameters():
        if param.requires_grad and adapter_name not in name:
            print(f"-- {name}: {param.numel():,}")
    # TODO: Find out why heads are added?


def show_houlsby_adapter():
    config = AdapterConfig.load("houlsby")
    print(f"\n ===== HOULSBY ADAPTER CONFIG =====")
    print(f"Architecture:      {config.__class__.__name__}")
    print(f"Reduction Factor:     {config['reduction_factor']}")
    print(f"Non-Linearity Activation:        {config['non_linearity']}")
    d_model = 768   # Default is 768 (see RobertaConfig documentation on huggingface)
    bottleneck_size = d_model // config["reduction_factor"]
    print(f"Bottlenecks reduction from {d_model} to {bottleneck_size}")


def show_adapter_injection():
    model = RobertaAdapterModel()
    intern_model = model.get_model_state()
    layer_0 = intern_model.roberta.encoder.layer[0]
    print(f"===== INSPECT ADAPTER INJECTION =====")
    print(f"Model output before: \n")
    print(layer_0.output)
    adapter_name = "adapter_0"
    model.add_adapter_layers(adapter_name)
    model = model.get_model_state()
    print("\n Model after adding one adapter:")
    layer_0 = model.roberta.encoder.layer[0]
    print(layer_0.output)


if __name__=="__main__":
    #show_parameter_stats()
    #show_houlsby_adapter()
    show_adapter_injection()
    # TODO: Sicherstellen, dass wirklich HOULSBY adapter verwendet wurde!
