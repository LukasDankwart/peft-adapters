from model.roberta import RobertaAdapterModel
from datasets import load_dataset, ClassLabel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer
import torch
import pandas as pd

def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class_id].item()

    return predicted_class_id, confidence

def seminar_comparison():

    """ Load seminar Adapter model """
    print(f"\n Load seminar adapter model...")
    wrapper = RobertaAdapterModel()
    adapter_model = wrapper.get_model_state()
    adapter_tokenizer = wrapper.tokenizer

    print(f"Load seminar adapters...")
    adapter_paths = {
        "seminar_adapter": "model/seminar_adapter/checkpoint-1270/seminar_adapter"
    }
    adapter_model.load_adapter(adapter_paths["seminar_adapter"])
    adapter_model.active_adapters = "seminar_adapter"
    adapter_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_model.to(device)

    """ Load seminar Baseline model """
    checkpoint_path = "model/baselines/seminar/checkpoint-1270"
    loaded_baseline_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=2
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    loaded_baseline_model.eval()
    loaded_baseline_model.to(device)

    """ load test data """
    test_data_path = "data/seminar/test/test.csv"
    pf = pd.read_csv(test_data_path, sep='\t', encoding='utf-8')

    """ Run inference"""
    adapter_precision = 0
    baseline_precision = 0
    for index, row in pf.iterrows():
        abstract = row['abstract']
        gt_label =row['label']

        # Adapter prediction
        adapter_predicted_class_id, confidence = predict(abstract, adapter_model, adapter_tokenizer, device)
        if adapter_predicted_class_id == gt_label:
            adapter_precision += 1

        # Baseline prediction
        baseline_predicted_class_id, confidence = predict(abstract, loaded_baseline_model, baseline_tokenizer, device)
        if baseline_predicted_class_id == gt_label:
            baseline_precision += 1

    adapter_model_params = sum(p.numel() for p in adapter_model.parameters())
    baseline_model_params = sum(p.numel() for p in loaded_baseline_model.parameters())
    print(f"Adapter precision: {adapter_precision / len(pf):.2f}%")
    print(f"-- Number of params: {adapter_model_params} \n")
    print(f"Baseline precision: {baseline_precision / len(pf):.2f}%")
    print(f"-- Number of params: {baseline_model_params} \n")
    print(f"Adapter-model has {adapter_model_params / baseline_model_params:.2f}% params of baseline.")


def sarcasm_comparison():
    def tokenize_sarcasm(example):
        return baseline_tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    """ Load sarcasm Adapter model """
    print(f"\n Load sarcasm adapter model...")
    wrapper = RobertaAdapterModel()
    adapter_model = wrapper.get_model_state()
    adapter_tokenizer = wrapper.tokenizer

    print(f"Load sarcasm adapters...")
    adapter_paths = {
        "sarcasm_adapter": "model/sarcasm_adapter/checkpoint-895/sarcasm_adapter"
    }
    adapter_model.load_adapter(adapter_paths["sarcasm_adapter"])
    adapter_model.active_adapters = "sarcasm_adapter"
    adapter_model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_model.to(device)

    """ Load sarcasm Baseline model """
    checkpoint_path = "model/baselines/sarcasm/checkpoint-895"
    loaded_baseline_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=2
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    loaded_baseline_model.eval()
    loaded_baseline_model.to(device)

    """ test data"""
    dataset = load_dataset("tweet_eval", "irony")
    dataset_tokenized = dataset.map(tokenize_sarcasm, batched=True)
    test_data = dataset_tokenized["validation"]

    """ Run inference"""
    adapter_precision = 0
    baseline_precision = 0
    for index, row in enumerate(test_data):
        sentence = row['text']
        gt_label = row['label']

        # Adapter prediction
        adapter_predicted_class_id, confidence = predict(sentence, adapter_model, adapter_tokenizer, device)
        if adapter_predicted_class_id == gt_label:
            adapter_precision += 1

        # Baseline prediction
        baseline_predicted_class_id, confidence = predict(sentence, loaded_baseline_model, baseline_tokenizer, device)
        if baseline_predicted_class_id == gt_label:
            baseline_precision += 1

    adapter_model_params = sum(p.numel() for p in adapter_model.parameters())
    baseline_model_params = sum(p.numel() for p in loaded_baseline_model.parameters())
    print(f"Adapter precision: {adapter_precision / len(test_data):.2f}%")
    print(f"-- Number of params: {adapter_model_params} \n")
    print(f"Baseline precision: {baseline_precision / len(test_data):.2f}%")
    print(f"-- Number of params: {baseline_model_params} \n")
    print(f"Adapter-model has {adapter_model_params / baseline_model_params:.2f}% params of baseline.")


if __name__ == "__main__":
    #seminar_comparison()
    sarcasm_comparison()