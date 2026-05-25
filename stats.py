from model.roberta import RobertaAdapterModel
from adapters import AdapterConfig
import matplotlib.pyplot as plt

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


def plot_loss_ablation():
    results = [
        {
            "name": "0.3",
            "losses": [
                0.698498547077179,
                0.6993235945701599,
                0.6902828216552734,
                0.6870179772377014,
                0.6874886155128479,
            ]
        },
        {
            "name": "0.1",
            "losses": [
                0.6263827085494995,
                0.5354766845703125,
                0.5422782301902771,
                0.4857971966266632,
                0.4609869420528412,
            ]
        },
        {
            "name": "0.075",
            "losses": [
                0.3769833445549011,
                0.3074946403503418,
                0.315958172082901,
                0.28308001160621643,
                0.28130096197128296,
            ]
        },
        {
            "name": "0.05",
            "losses": [
                0.2516227960586548,
                0.2562691867351532,
                0.28681695461273193,
                0.26869112253189087,
                0.27379053831100464,
            ]
        },
        {
            "name": "no_adapter",
            "losses": [
                0.6408296823501587,
                0.5993636250495911,
                0.5622190833091736,
                0.542702436447143,
                0.5373716354370117,
            ]
        }
    ]

    epochs = [1, 2, 3, 4, 5]
    plt.figure(figsize=(8, 5))
    for result in results:
        name = result["name"]
        if name == "no_adapter":
            name = "- No Adapter"
        else:
            name = f"Std=" + name
        losses = result["losses"]
        print(name)
        print(losses)
        plt.plot(epochs, losses, marker='o', label=name)
    plt.xlabel("Epochs")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss of different parameters initializations")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(f"loss_ablation.png", dpi=400)




if __name__=="__main__":
    #show_parameter_stats()
    #show_houlsby_adapter()
    #show_adapter_injection()
    plot_loss_ablation()
