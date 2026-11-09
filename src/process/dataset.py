
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.config.config import *
from torch.utils.data import DataLoader

def get_dataset(ds_type):
    path=str(PROCESSED_DATA_DIR/ds_type)
    dataset=load_from_disk(path)
    return dataset

def dataloader(tokenizer,ds_type):
    dataset=get_dataset(ds_type)

    #定义数据整理器
    collator=DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
    )
    dataloader=DataLoader(dataset=dataset,batch_size=BATCH_SIZE,collate_fn=collator,shuffle=True)
    return dataloader

if __name__ == '__main__':
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    dataloader=dataloader(tokenizer,ds_type="train")
    for batch in dataloader:
        for k,v in batch.items():
            print(k,"->",v.shape)
            print(v)
        break