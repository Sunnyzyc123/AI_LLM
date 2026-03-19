# 本地文档 / 网页 RAG 问答系统
基于 LangChain 和 Gradio 的检索增强生成（RAG）问答系统，支持上传本地文档或输入网页 URL，使用 DeepSeek 大模型进行智能问答。

## ✨ 特性

- 📄 **多格式文档支持**：支持 PDF、DOCX、TXT、HTML 格式文档
- 🌐 **网页内容抓取**：支持直接输入网页 URL 进行问答
- 🎯 **智能检索**：基于本地嵌入模型的语义检索
- 💬 **多种 Prompt 模板**：提供多种预设 Prompt 模板，支持自定义 Prompt
- 🎨 **友好界面**：基于 Gradio 的现代化 Web 界面
- 🔒 **本地部署**：嵌入模型本地运行，保护数据隐私

## 📋 系统要求

- Python 3.8 或更高版本
- 操作系统：Windows / Linux / macOS

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/Sunnyzyc123/AI_LLM.git
cd RAG_L2
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备嵌入模型

下载 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 模型，并将其放置在 `./models/all-MiniLM-L6-v2` 目录下。

目录结构应如下：
```
RAG_L2/
├── models/
│   └── all-MiniLM-L6-v2/
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
├── RAG_L2.py
├── requirements.txt
└── README.md
```

### 4. 配置 API Key

设置 DeepSeek API Key 环境变量：

**Windows:**
```bash
set DEEPSEEK_API_KEY=你的密钥
```

**Linux/macOS:**
```bash
export DEEPSEEK_API_KEY=你的密钥
```

获取 API Key 请访问：[DeepSeek 官网](https://platform.deepseek.com/)

### 5. 运行程序

```bash
python RAG_L2.py
```

程序将自动在浏览器中打开 Web 界面（默认端口 7861）。

## 📖 使用说明

### 基本使用

1. 上传本地文件，或输入网页 URL
2. 选择预设的 Prompt 模板或输入自定义 Prompt
3. 输入你的问题并点击提交
4. 程序会先检索相关内容，再调用 DeepSeek 生成回答

### Prompt 模板说明

系统提供以下预设 Prompt 模板：

- **默认**：标准问答模板
- **详细解释**：提供全面详细的解答
- **简洁总结**：突出关键信息
- **技术文档**：专业的技术文档分析
- **学术分析**：严谨的学术风格回答

### 自定义 Prompt

你也可以使用自定义 Prompt 模板，使用 `{context}` 和 `{question}` 作为占位符：
```
请根据以下上下文信息回答问题。
上下文：{context}
问题：{question}
```

## 🔧 配置说明

### 环境变量

- `DEEPSEEK_API_KEY`：DeepSeek API 密钥（必需）
- `GRADIO_SERVER_NAME`：服务器地址（默认：127.0.0.1）
- `GRADIO_SERVER_PORT`：服务器端口（默认：7861）

### 端口配置

如果默认端口被占用，程序会自动尝试使用后续端口（7861-7880）。

## 📁 项目结构

```
RAG_L2/
├── models/                    # 本地模型目录
│   └── all-MiniLM-L6-v2/     # 嵌入模型
├── RAG_L2.py                 # 主程序文件
├── requirements.txt          # 依赖包列表
└── README.md                # 项目说明文档
```

## 🛠️ 技术栈

- **LangChain**：LLM 应用框架
- **Gradio**：Web 界面框架
- **Chroma**：向量数据库
- **DeepSeek**：大语言模型
- **HuggingFace Transformers**：嵌入模型

