# 关于项目
这是一个利用大语言模型缩写文本的项目，可用来将太多字数的作文缩写到不够字数以展现自己的 fw 属性（？）

## 环境配置

需要提前装好：
1. anaconda / miniconda
2. cuda 12.6 （可选，提升速度）

在项目目录下运行以下命令
```bash
conda create -n nlp python==3.12
conda activate nlp
pip install -r requirements.txt
```

## 准备数据集
1. 将训练数据集填充到 `train.csv` ，需要包含两列：
   - text：未经压缩的原始数据
   - summary：压缩过的数据

2. 将测试数据集填充到 `generate.csv` ，需要包含一列：
   - input：需要被压缩的原始数据

## 修改配置文件
根据需要修改 `config.ini` 文件

## 训练模型
```bash
python 00-train.py
```
完成后会生成模型目录，此时可配置 `config.ini` 继续上次的训练。

## 生成数据
```bash
python 01-generate.py
```
完成后会生成 `generate_with_output.csv` ，压缩后的数据会存放在 `output` 列中。