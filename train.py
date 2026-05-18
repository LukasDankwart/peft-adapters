from datasets import load_dataset, ClassLabel
from transformers import TrainingArguments
from adapters import AdapterTrainer

from model.roberta import RobertaAdapterModel
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

"""
    Code for training the seminar adapter
"""

def train_seminar_adapter(model, adapter_name="seminar_adapter"):
    # Prepare model (adapters) for training
    intern_model = model.get_model_state()
    model.add_adapter_layers(adapter_name=adapter_name)
    model.add_classification_head(head_name=adapter_name)
    model.prepare_for_training(adapter_name)
    intern_model.active_adapters = adapter_name

    def tokenize_seminar(example):
        return model.tokenizer(
            example["abstract"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Load and prepare train data
    path_to_train_csv = "data/seminar/train/train.csv"
    train_dataset = load_dataset("csv", data_files=path_to_train_csv, sep="\t")
    train_data_tokenized = train_dataset.map(tokenize_seminar, batched=True)

    # Load and prepare test data
    path_to_test_csv = "data/seminar/test/test.csv"
    test_dataset = load_dataset("csv", data_files=path_to_test_csv, sep="\t")
    test_data_tokenized = test_dataset.map(tokenize_seminar, batched=True)
    print(f"[DONE] Loading train and test data.")

    # Configure train parameters
    training_args = TrainingArguments(
        output_dir = "model/seminar_adapter",
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = AdapterTrainer(
        model=model.get_model_state(),
        args=training_args,
        train_dataset=train_data_tokenized["train"],
        eval_dataset=test_data_tokenized["train"],
    )

    print(f"[START] Training model (adapter={adapter_name}) ...")
    trainer.train()
    print(f"[DONE] Training model (adapter={adapter_name}) finished.")


def seminar_data_preprocessing(file_path, train_dir="data/seminar/train", test_dir="data/seminar/test"):
    # Load and split dataset
    raw_dataset = load_dataset("csv", data_files=file_path, sep="\t")
    new_features = raw_dataset["train"].features.copy()
    new_features["label"] = ClassLabel(num_classes=2, names=["irrelevant", "relevant"])
    raw_dataset["train"] = raw_dataset["train"].cast(new_features)
    split_dataset = raw_dataset["train"].train_test_split(
        test_size=0.1,
        stratify_by_column="label",
        seed=42,
    )
    # Save train and test data
    split_dataset["train"].to_csv(train_dir + "/train.csv", sep="\t", index=False)
    split_dataset["test"].to_csv(test_dir + "/test.csv", sep="\t", index=False)
    print(f"[DONE] Preprocessing seminar data done! Training: {len(split_dataset['train'])}, test: {len(split_dataset['test'])} ...")


"""
    Code for training the sacarsm adapter
"""

def train_sarcasm_adapter(model, adapter_name="sarcasm_adapter"):
    # Prepare model (adapters) for training
    intern_model = model.get_model_state()
    model.add_adapter_layers(adapter_name=adapter_name)
    model.add_classification_head(head_name=adapter_name)
    model.prepare_for_training(adapter_name)
    intern_model.active_adapters = adapter_name

    def tokenize_sarcasm(example):
        return model.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Load and prepare train data
    dataset = load_dataset("tweet_eval", "irony")
    dataset_tokenized = dataset.map(tokenize_sarcasm, batched=True)

    # Configure train parameters
    training_args = TrainingArguments(
        output_dir="model/sarcasm_adapter",
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = AdapterTrainer(
        model=model.get_model_state(),
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
    )
    print(f"[START] Training model (adapter={adapter_name}) ...")
    trainer.train()
    print(f"[DONE] Training model (adapter={adapter_name}) finished.")

"""
    Train baseline for seminar (pretrained BERT + class head, no adapters!)
"""
def train_seminar_baseline():
    # Prepare model (adapters) for training
    baseline_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    for param in baseline_model.roberta.parameters():
        param.requires_grad = False
    num_train_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

    def tokenize_seminar(example):
        return model.tokenizer(
            example["abstract"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Load and prepare train data
    path_to_train_csv = "data/seminar/train/train.csv"
    train_dataset = load_dataset("csv", data_files=path_to_train_csv, sep="\t")
    train_data_tokenized = train_dataset.map(tokenize_seminar, batched=True)

    # Load and prepare test data
    path_to_test_csv = "data/seminar/test/test.csv"
    test_dataset = load_dataset("csv", data_files=path_to_test_csv, sep="\t")
    test_data_tokenized = test_dataset.map(tokenize_seminar, batched=True)
    print(f"[DONE] Loading train and test data.")

    # Configure train parameters
    training_args = TrainingArguments(
        output_dir="model/baselines/seminar",
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=baseline_model,
        args=training_args,
        train_dataset=train_data_tokenized["train"],
        eval_dataset=test_data_tokenized["train"],
    )

    print(f"[START] Training seminar-baseline model (class head)...")
    print(f"-- Number of trainable params: {num_train_params}")
    trainer.train()
    print(f"[DONE] Training model seminar-baseline model finished.")


"""
    Train baseline for sarcasm (pretrained BERT + class head, no adapters!)
"""
def train_sarcasm_baseline():
    # Prepare model (adapters) for training
    baseline_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    for param in baseline_model.roberta.parameters():
        param.requires_grad = False
    num_train_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)

    def tokenize_sarcasm(example):
        return model.tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    # Load and prepare train data
    dataset = load_dataset("tweet_eval", "irony")
    dataset_tokenized = dataset.map(tokenize_sarcasm, batched=True)

    # Configure train parameters
    training_args = TrainingArguments(
        output_dir="model/baselines/sarcasm",
        learning_rate=1e-4,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=baseline_model,
        args=training_args,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
    )

    print(f"[START] Training sarcasm-baseline model (class head)...")
    print(f"-- Number of trainable params: {num_train_params}")
    trainer.train()
    print(f"[DONE] Training model seminar-baseline model finished.")


if __name__ == "__main__":

    """ Code for training the seminar adapter
    """
    #seminar_data_preprocessing("data/seminar/seminar.csv")

    model = RobertaAdapterModel()

    """ Train seminar adapter variant
    train_seminar_adapter(model)
    """

    """ Train sarcasm adapter variant
    train_sarcasm_adapter(model)
    """

    """ Train seminar BASELINE
    train_seminar_baseline()
    """

    """ Train seminar BASELINE 
    """
    train_sarcasm_baseline()
