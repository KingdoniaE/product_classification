from dataclasses import dataclass
import time
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from src.process.dataset import *
from src.config.config import *
import torch


#定义训练配置类
@dataclass
class TrainConfig:
    epochs: int=EPOCHS
    output_dir: str=MODELS_DIR
    learning_rate: float=LEARNING_RATE
    batch_size: int=BATCH_SIZE
    save_steps: int=SAVE_STEPS

    output_dir: str = MODELS_DIR
    log_dir: str = LOGS_DIR

    early_stop_patience: int =EARLY_STOPP_PATIENCE
    early_stop_metrics:str='loss'
    use_amp:bool=True

#定义训练器类
class Trainer:
    def __init__(self,model,train_dataset,valid_dataset,collator,compute_metrics,device,config):
        self.device=device
        self.train_dataset=train_dataset
        self.valid_dataset=valid_dataset
        self.model=model.to(self.device)
        self.config=config
        self.collator=collator
        self.compute_metrics=compute_metrics
        self.checkpoint_path=Path(self.config.output_dir)/'last'/'checkpoint.pt'

        self.scaler=torch.amp.GradScaler(device=self.device,enabled=config.use_amp)
        #定义优化器
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.step=0
        self.early_stop_best_score=-float('inf')
        self.early_stop_counter=0

        self.writer = SummaryWriter(log_dir=str(Path(self.config.log_dir)/time.strftime('%Y%m%d_%H%M%S')))
    def _get_dataloader(self,dataset):
        dataloader=DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.collator,
            shuffle=True
        )
        return dataloader
    def _train_one_batch(self,batch):
        self.model.train()
        batch={k:v.to(self.device) for k,v in batch.items() }

        with torch.autocast(
                enabled=self.config.use_amp,
                device_type=self.device.type,
                dtype=torch.float16
        ):
            outputs=self.model(**batch)
        self.scaler.scale(outputs.loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # outputs=self.model(**batch)
        # outputs.loss.backward()
        # self.optimizer.step()

        self.optimizer.zero_grad()
        return outputs.loss.item()
    def _save_checkpoint(self):
        checkpoint={
            'model': self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scaler':self.scaler.state_dict(),
            'step':self.step,
            'early_stop_best_score':self.early_stop_best_score,
            'early_stop_counter':self.early_stop_counter,
        }
        tqdm.write(f'Save checkpoint ....')
        torch.save(checkpoint,self.checkpoint_path)
    def _load_checkpoint(self):
        if self.checkpoint_path.exists():
            print('发现检查点，继续训练...')
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])
            self.step = checkpoint['step']
            self.early_stop_best_score = checkpoint['early_stop_best_score']
            self.early_stop_counter = checkpoint['early_stop_counter']
        else:
            print('没有发现检查点，从头开始训练')
    def _should_early_stop(self,metrics):
        metric=metrics[self.config.early_stop_metrics]
        score=-metric if self.config.early_stop_metrics=='loss' else metric
        if score>self.early_stop_best_score:
            self.early_stop_best_score=score
            self.early_stop_counter=0
            tqdm.write('save best...')
            self.model.save_pretrained(Path(self.config.output_dir)/'best')
            return False
        else:
            self.early_stop_counter+=1
            if self.early_stop_counter>=self.config.early_stop_patience:
                return True
            else:
                return False

    def train(self):
        self._load_checkpoint()

        dataloader = self._get_dataloader(self.train_dataset)

        for epoch in range(self.config.epochs):
            for batch in tqdm(dataloader,desc=f'Epoch {epoch+1}'):
                this_loss=self._train_one_batch(batch)
                self.step+=1
                if self.step%self.config.save_steps==0:
                    tqdm.write(f'[Epoch:{epoch+1} | Step:{self.step}] Loss:{this_loss:.4f}')
                    self.writer.add_scalar('Loss/train',this_loss,self.step)

                    metrics = self.evaluate()
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f"[Evaluate] {metrics_str}")

                    if self._should_early_stop(metrics):
                        tqdm.write('stop!!!!!')
                        return
                    self._save_checkpoint()

                    # if metrics['loss']<self.min_loss:
                    #     self.min_loss=metrics['loss']
                    #     tqdm.write('save model....')
                    #     # self.model.save_pretrained(self.config.output_dir)
    def evaluate(self )-> dict:
        dataloader = self._get_dataloader(self.valid_dataset)

        self.model.eval()
        total_loss=0
        predictions=[]
        labels=[]
        with torch.no_grad():
            for batch in tqdm(dataloader,desc='验证'):
                batch={k:v.to(self.device) for k,v in batch.items() }
                outputs=self.model(**batch)
                loss=outputs.loss.item()
                total_loss+=loss
                this_prediction=torch.argmax(outputs.logits,dim=-1)
                predictions.extend(this_prediction.tolist())
                this_labels=batch['labels']
                labels.extend(this_labels.tolist())
        loss=total_loss/len(dataloader)
        metrics=self.compute_metrics(predictions,labels)
        return {'loss': loss, **metrics}

def train():
    print("Training...")
    #定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #定义分词器
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    #加载标签
    with open(MODELS_DIR/LABELS_FILE,'r', encoding='utf-8') as f:
        all_labels = f.read().splitlines()

    # train_dataset = get_dataset(ds_type='train')
    # valid_dataset = get_dataset(ds_type='valid')
    # print("训练集：", train_dataset)
    # print("验证集：", valid_dataset)
    # print("字段名称：", train_dataset.column_names)
    # print("训练集第一条数据：")
    # print(train_dataset[0])
    # print("验证集第一条数据：")
    # print(valid_dataset[0])

    #加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(all_labels),
        id2label={i:label for i,label in enumerate(all_labels)},
        label2id={label:i for i,label in enumerate(all_labels)}
    ).to(device)

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
    train_dataset=get_dataset(ds_type='train').select(range(100))
    valid_dataset = get_dataset(ds_type='valid').select(range(100))
    #
    # from collections import Counter
    # train_dataset = get_dataset(ds_type='train')
    # valid_dataset = get_dataset(ds_type='valid')
    # print("类别总数：", len(all_labels))
    # print("\n训练集标签分布：")
    # print(Counter(train_dataset['labels']))
    # print("\n验证集标签分布：")
    # print(Counter(valid_dataset['labels']))
    # print("\n训练集标签范围：")
    # print(min(train_dataset['labels']), max(train_dataset['labels']))
    # print("\n验证集标签范围：")
    # print(min(valid_dataset['labels']), max(valid_dataset['labels']))

    #定义训练器
    trainer=Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collator=collator,
        compute_metrics=compute_metrics,
        device=device,
        config=TrainConfig()
    )
    #开始训练
    trainer.train()

if __name__ == '__main__':
    train()