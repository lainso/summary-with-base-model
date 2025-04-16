import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import configparser
import os

class SummaryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_input_length, max_target_length):
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data.iloc[idx]['text']
        summary = self.data.iloc[idx]['summary']

        inputs = self.tokenizer.encode_plus(
            article,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer.encode_plus(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def read_config():
    config = configparser.ConfigParser()
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)
    return config

def main():
    print("别急，在跑了，给爷等会。。。")
    config = read_config()
    
    # 通用参数
    model_name = config['common']['model_name']
    model_dir = config['common']['model_dir']
    input_max_length = int(config['common']['input_max_length'])
    output_max_length = int(config['common']['output_max_length'])
    
    # 训练参数
    train_data_path = config['training']['train_data_path']
    batch_size = int(config['training']['batch_size'])
    epochs = int(config['training']['epochs'])
    learning_rate = float(config['training']['learning_rate'])

    # 初始化分词器和模型
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 创建数据集和数据加载器
    train_dataset = SummaryDataset(
        data_path=train_data_path,
        tokenizer=tokenizer,
        max_input_length=input_max_length,
        max_target_length=output_max_length
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 保存模型和分词器
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    if not os.path.exists('config.ini'):
        print("byd你 config.ini 去哪了？")
        exit(1)
    main()