import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from src.config.config import *
from sklearn.metrics import accuracy_score, f1_score, recall_score
from src.process.dataset import get_dataset
from src.runner.train import Trainer,TrainConfig
def evaluate():
    print("开始评估...")
    #定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #定义分词器
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    #加载模型
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR/'best',).to(device)

    #定义数据整理器
    collator=DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )

    def compute_metrics(predictions,labels)-> dict:
        accuracy=accuracy_score(labels,predictions)
        f1=f1_score(labels,predictions,average='weighted')
        recall=recall_score(labels,predictions,average='weighted')
        return {'accuracy':accuracy,'recall':recall,'f1':f1}

    #定义数据加载器
    test_dataset = get_dataset(ds_type='test').select(range(200))
    #定义训练器
    trainer=Trainer(
        model=model,
        train_dataset=None,
        valid_dataset=test_dataset,
        collator=collator,
        compute_metrics=compute_metrics,
        device=device,
        config=TrainConfig()
    )
    #开始训练
    metrics=trainer.evaluate()
    print(f"评估结果:{metrics}")
if __name__ == '__main__':
    evaluate()