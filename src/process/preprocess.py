from datasets import load_dataset
from transformers import AutoTokenizer

from src.config.config import *
def preprocess():
    print("Preprocessing... ")
    dataset_dict=load_dataset('csv',data_files={
        'train': str(RAW_DATA_DIR/TRAIN_DATA_FILE),
        'test': str(RAW_DATA_DIR/TEST_DATA_FILE),
        'valid': str(RAW_DATA_DIR/VALID_DATA_FILE)
    },delimiter='\t')
    # print(dataset_dict)

    #加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    #编码文本，得到模型输入
    def batch_encode(batch):
        inputs=tokenizer(batch['text_a'],truncation=True)
        inputs['labels']=batch['label']
        return inputs
    dataset_dict=dataset_dict.map(batch_encode,batched=True,remove_columns=['label','text_a'])
    print(dataset_dict['train'][:3])
if __name__ == "__main__":
    preprocess()