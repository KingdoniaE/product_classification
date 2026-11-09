from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config.config import *

class Predict:
    def __init__(self,model,tokenizer,device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self,text:str | list[str]):
        is_str=isinstance(text,str)
        if is_str:
            text=[text]
        inputs=self.tokenizer(text,padding=True,truncation=True,return_tensors="pt")
        inputs={k: v.to(self.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits,dim=-1).tolist()
        predictions_labels=[self.model.config.id2label[i] for i in predictions]
        if is_str:
            return predictions_labels[0]
        return predictions_labels
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR/'best')
    predictor = Predict(model,tokenizer,device)
    text = '好奇心钻装纸尿裤L40片9-14kg'
    print(predictor.predict(text))

    texts = ['911-267遥控车', '瓦伦丁Wurenbacher小麦西柚啤酒500ml*12听整箱装德国原装进口果啤', '640G正航牛奶早餐饼干']
    print(predictor.predict(texts))
if __name__ == '__main__':
    predict()
