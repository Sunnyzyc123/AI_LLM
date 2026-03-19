# 这个代码使用gradio搭建了一个可视化界面，可以上传文档或url，然后进行提问和回答


import os
import socket
import warnings
from pathlib import Path
from langchain.prompts import PromptTemplate

os.environ.setdefault("USER_AGENT", "RAG_L2/1.0")

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

    warnings.filterwarnings(
        "ignore",
        message=".*HuggingFaceEmbeddings.*deprecated.*",
    )

BASE_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "请先设置环境变量 DEEPSEEK_API_KEY，例如：set DEEPSEEK_API_KEY=你的密钥"
    )

if not LOCAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"本地嵌入模型目录不存在：{LOCAL_MODEL_PATH}\n"
        "请确认模型文件已放到 ./models/all-MiniLM-L6-v2 下。"
    )


def create_llm():
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=1024,
    )


embedding_model = HuggingFaceEmbeddings(
    model_name=str(LOCAL_MODEL_PATH),
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5000,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "，", " ", "",",","."],
)


def load_uploaded_file(file_obj):
    file_name = file_obj.name
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext == ".pdf":
        loader = PyPDFLoader(file_name)
    elif file_ext == ".docx":
        loader = Docx2txtLoader(file_name)
    elif file_ext == ".txt":
        loader = TextLoader(file_name, encoding="utf-8")
    elif file_ext in (".html", ".htm"):
        loader = UnstructuredHTMLLoader(file_name)
    else:
        raise ValueError(
            f"不支持的文件类型：{file_ext}，目前仅支持 PDF、DOCX、TXT、HTML。"
        )

    return loader.load()


# 预设的Prompt模板
PROMPT_TEMPLATES = {
    "默认": "请根据以下上下文信息回答问题。如果上下文中没有相关信息，请说明无法从提供的材料中找到答案。\n\n上下文：{context}\n\n问题：{question}",
    
    "详细解释": "你是一位专业的文档分析师。请基于提供的上下文信息，对问题进行详细、全面的解答。回答应包括：1)直接回答 2)相关背景 3)补充说明\n\n上下文：{context}\n\n问题：{question}",
    
    "简洁总结": "请用简洁明了的语言总结回答问题，突出关键信息，避免冗长描述。\n\n上下文：{context}\n\n问题：{question}",
    
    "技术文档": "作为技术文档专家，请基于以下技术文档内容回答问题。回答应专业、准确，必要时包含技术细节。\n\n上下文：{context}\n\n问题：{question}",
    
    "学术分析": "请以学术研究的角度分析并回答问题，引用提供的上下文内容，保持客观严谨的学术风格。\n\n上下文：{context}\n\n问题：{question}"
}

def process_input(file_obj, url, question,prompt_template=None):
    if file_obj is None and not url.strip():
        return "请上传文件或输入 URL。", ""
    if not question.strip():
        return "请输入问题。", ""

    documents = []

    if file_obj is not None:
        try:
            documents.extend(load_uploaded_file(file_obj))
        except Exception as exc:
            return f"加载文件失败：{exc}", ""

    if url.strip():
        try:
            loader = WebBaseLoader(url.strip())
            documents.extend(loader.load())
        except Exception as exc:
            return f"加载 URL 失败：{exc}", ""

    if not documents:
        return "未能从文件或 URL 中提取到内容。", ""

    chunks = text_splitter.split_documents(documents)
    if not chunks:
        return "文档切分后没有可用内容。", ""

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 使用自定义Prompt或默认Prompt
    if prompt_template:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
    else:
        prompt = None

    qa_chain = RetrievalQA.from_chain_type(
        llm=create_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt} if prompt else {},
        return_source_documents=True,
        verbose=False,
    )

    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = "\n\n".join(
        doc.page_content for doc in result["source_documents"]
    )

    return answer, sources






with gr.Blocks(title="本地文档 / 网页 RAG 问答", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 上传文档或输入网页 URL，基于内容进行问答\n"
        "支持 `PDF / DOCX / TXT / HTML`，嵌入模型从本地目录加载。"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传文件（可选）",
                file_types=[".pdf", ".docx", ".txt", ".html", ".htm"],
                file_count="single",
            )
            url_input = gr.Textbox(
                label="网页 URL（可选）",
                placeholder="例如：https://example.com/article.html",
                lines=1,
            )

            # 添加Prompt模板选择器
            prompt_selector = gr.Dropdown(
                label="选择Prompt模板",
                choices=list(PROMPT_TEMPLATES.keys()),
                value="默认",
                interactive=True
            )
            
            # 添加自定义Prompt输入框
            custom_prompt = gr.Textbox(
                label="自定义Prompt模板（可选）",
                placeholder="输入自定义Prompt模板，使用 {context} 和 {question} 作为占位符",
                lines=4,
                interactive=True
            )

            question_input = gr.Textbox(
                label="你的问题",
                placeholder="例如：这份文档的核心内容是什么？",
                lines=2,
            )
            submit_btn = gr.Button("提交问题", variant="primary")

        with gr.Column(scale=1):
            answer_output = gr.Textbox(label="回答", lines=10, interactive=False)
            sources_output = gr.Textbox(label="检索到的参考片段", lines=8, interactive=False)

    # 添加Prompt模板更新逻辑
    def update_prompt_template(selected_template):
        return PROMPT_TEMPLATES[selected_template]

    prompt_selector.change(
        fn=update_prompt_template,
        inputs=[prompt_selector],
        outputs=[custom_prompt]
    )

    submit_btn.click(
        fn=process_input,
        inputs=[file_input, url_input, question_input],
        outputs=[answer_output, sources_output],
        api_name=False,
        show_api=False,
        queue=False,
    )

    gr.Markdown(
        "---\n"
        "### 使用说明\n"
        "1. 上传本地文件，或输入网页 URL。\n"
        "2. 选择预设的Prompt模板或输入自定义Prompt。\n"
        "3. 输入你的问题并点击提交。\n"
        "4. 程序会先检索相关内容，再调用 DeepSeek 生成回答。\n\n"
        "### Prompt模板说明\n"
        "- 默认：标准问答模板\n"
        "- 详细解释：提供全面详细的解答\n"
        "- 简洁总结：突出关键信息\n"
        "- 技术文档：专业的技术文档分析\n"
        "- 学术分析：严谨的学术风格回答"
    )



def find_available_port(preferred_port=7861, host="127.0.0.1", max_tries=20):
    for port in range(preferred_port, preferred_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"从端口 {preferred_port} 开始未找到可用端口。")



def launch_demo():
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    bind_host = "0.0.0.0" if server_name == "0.0.0.0" else "127.0.0.1"
    server_port = find_available_port(preferred_port=preferred_port, host=bind_host)

    launch_kwargs = {
        "server_name": server_name,
        "server_port": server_port,
        "inbrowser": True,
        "show_api": False,
    }

    print(f"Gradio 启动端口：{server_port}")

    try:
        demo.launch(**launch_kwargs)
    except ValueError as exc:
        if "localhost is not accessible" not in str(exc):
            raise
        print("检测到 localhost 不可访问，自动切换为 share=True。")
        demo.launch(**launch_kwargs, share=True)


if __name__ == "__main__":
    launch_demo()
