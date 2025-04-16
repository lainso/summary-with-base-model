import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import configparser
import os

def generate_summary(text, model, tokenizer, device, max_length, num_beams):
    model.eval()
    inputs = tokenizer.encode(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def read_config():
    config = configparser.ConfigParser()
    with open('config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)
    return config

def main():
    config = read_config()
    
    # 通用参数
    model_dir = config['common']['model_dir']
    input_max_length = int(config['common']['input_max_length'])
    output_max_length = int(config['common']['output_max_length'])
    
    # 生成参数
    generate_data_path = config['generate']['generate_data_path']
    output_csv = config['generate']['output_csv']
    generate_batch_size = int(config['generate']['generate_batch_size'])
    num_beams = int(config['generate']['num_beams'])

    # 加载模型和分词器
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 读取输入数据
    df = pd.read_csv(generate_data_path)
    texts = df['input'].tolist()

    # 生成摘要
    summaries = []
    for i in range(0, len(texts), generate_batch_size):
        batch_texts = texts[i:i+generate_batch_size]
        batch_summaries = [
            generate_summary(
                text,
                model,
                tokenizer,
                device,
                max_length=output_max_length,
                num_beams=num_beams
            )
            for text in batch_texts
        ]
        summaries.extend(batch_summaries)
    
    # 将结果保存到DataFrame
    df['output'] = summaries
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    if not os.path.exists('config.ini'):
        print("Error: config.ini not found!")
        exit(1)
    main()