import os
import torch
import pandas as pd
from model.roberta import RobertaAdapterModel


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


def run_inference(papers):
    print(f"\n Load base model...")
    wrapper = RobertaAdapterModel()
    model = wrapper.get_model_state()
    tokenizer = wrapper.tokenizer

    print(f"Load adapters...")
    adapter_paths = {
        "seminar_adapter": "model/seminar_adapter/checkpoint-1270/seminar_adapter"
    }

    model.load_adapter(adapter_paths["seminar_adapter"])
    model.active_adapters = "seminar_adapter"
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"[START] Starting inference for given papers...")
    for paper in papers:
        title = paper["title"]
        if not paper_in_data(paper):
            print(f"\n [START] Starting classification of paper: '{title}' ...")
            predicted_class_id, confidence = predict(paper["abstract"], model, tokenizer, device)
            if predicted_class_id == 1:
                print(f"✅ Prof. Eggensperger would like this!")
            else:
                print(f"❌ Given paper seems irrelevant for our seminar.")
        else:
            print(f"\n ℹ️ Paper'{title}' is included in train/test data! Inference is skipped!")


def paper_in_data(paper, csv_path="data/seminar/seminar.csv"):
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Given path {csv_path} does not exist")
    pf = pd.read_csv(csv_path, sep='\t')
    titles = pf["title"]
    titles_normalized = titles.str.lower().str.strip()
    search_title = paper["title"].lower().strip()
    if search_title in titles_normalized.values:
        return True
    return False


if __name__ == "__main__":
    papers = [
        {
            "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
            "abstract": "There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU."
        },
        {
            "title": "Parameter-Efficient Transfer Learning for NLP",
            "abstract": "Fine-tuning large pre-trained models is an effective transfer mechanism in NLP. However, in the presence of many downstream tasks, fine-tuning is parameter inefficient: an entire new model is required for every task. As an alternative, we propose transfer with adapter modules. Adapter modules yield a compact and extensible model; they add only a few trainable parameters per task, and new tasks can be added without revisiting previous ones. The parameters of the original network remain fixed, yielding a high degree of parameter sharing. To demonstrate adapter's effectiveness, we transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters attain near state-of-the-art performance, whilst adding only a few parameters per task. On GLUE, we attain within 0.4% of the performance of full fine-tuning, adding only 3.6% parameters per task. By contrast, fine-tuning trains 100% of the parameters per task."
        },
        {
            "title": "Stroke-Based Cursive Character Recognition",
            "abstract": "Human eye can see and read what is written or displayed either in natural handwriting or in printed format. The same work in case the machine does is called handwriting recognition. Handwriting recognition can be broken down into two categories: off-line and on-line. ..."
        },
        {
            "title": "LoRA: Low-Rank Adaptation of Large Language Models",
            "abstract": "An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency. We also provide an empirical investigation into rank-deficiency in language model adaptation, which sheds light on the efficacy of LoRA. We release a package that facilitates the integration of LoRA with PyTorch models and provide our implementations and model checkpoints for RoBERTa, DeBERTa, and GPT-2 at"
        }
    ]
    run_inference(papers)