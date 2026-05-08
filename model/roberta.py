from transformers import AutoTokenizer
from adapters import AutoAdapterModel


class RobertaAdapterModel:

    def __init__(self):
        self.model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoAdapterModel.from_pretrained(self.model_name)


    def add_adapter_layers(self, adapter_name):
        if self.model is not None:
            self.model.add_adapter(adapter_name, config="houlsby")
        raise RuntimeError(f"Model state is none")


    def add_classification_head(self, head_name, num_labels=2):
        if self.model is not None:
            self.model.add_classification_head(head_name, num_labels=num_labels)
        raise RuntimeError(f"Model state is none")


    def get_model_state(self):
        if self.model is not None :
            return self.model
        raise RuntimeError(f"Model state is none")
