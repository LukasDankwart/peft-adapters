from transformers import AutoTokenizer
from adapters import AutoAdapterModel


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