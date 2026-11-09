
from datasets import load_dataset
from transformers import AutoTokenizer
# 导入项目配置：数据路径、模型名称、文件名称等
from src.config.config import *

def preprocess():
    print("Preprocessing... ")
    # 1. 加载原始数据集
    dataset_dict = load_dataset('csv', data_files={
        'train': str(RAW_DATA_DIR / TRAIN_DATA_FILE),
        'test': str(RAW_DATA_DIR / TEST_DATA_FILE),
        'valid': str(RAW_DATA_DIR / VALID_DATA_FILE)
    }, delimiter='\t')
    # 2. 获取所有分类标签
    all_labels = sorted(set(dataset_dict['train']['label']))
    print(all_labels)
    with open(MODELS_DIR / LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_labels))
    # 3. 将字符串标签编码成整数
    dataset_dict = dataset_dict.class_encode_column('label')
    # 4. 加载预训练模型对应的分词器
    # 根据 MODEL_NAME 加载对应的 Tokenizer
    # Tokenizer 负责将文本转换成模型可以处理的数字序列
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 5. 定义批量文本编码函数
    def batch_encode(batch):
        inputs = tokenizer(batch['text_a'], truncation=True)
        inputs['labels'] = batch['label']
        return inputs
    # 6. 对整个数据集进行文本编码
    dataset_dict = dataset_dict.map(
        batch_encode,
        batched=True,
        remove_columns=['label', 'text_a']
    )
    # 7. 查看处理后的数据
    print(dataset_dict['train'][:3])
    # 8. 保存预处理后的数据集
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)
if __name__ == "__main__":
    preprocess()