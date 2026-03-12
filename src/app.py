# app.py
import streamlit as st
import requests
import json
from datetime import datetime

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="汽车电子知识库助手",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义 CSS 样式 ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.header("⚙️ 系统设置")

    # API 配置
    API_URL = st.text_input("API 地址", value="http://localhost:8000")

    # 模式选择
    st.subheader("🎯 问答模式")
    query_mode = st.radio(
        "选择问答模式",
        ["RAG 模式", "Agent 模式"],
        index=0,
        help="RAG: 只查知识库\nAgent: 自动判断使用知识库/搜索/计算器"
    )

    # 检索参数
    st.subheader("🔍 检索配置")
    top_k = st.slider("检索文档数量", 1, 10, 3)

    # RAG 增强配置
    st.subheader("🔧 RAG 增强")

    # 初始化 session state
    if "rag_config" not in st.session_state:
        st.session_state.rag_config = {"enable_hyde": True, "enable_query_expansion": True}

    # 尝试获取服务器配置
    try:
        config_response = requests.get(f"{API_URL}/config", timeout=5)
        if config_response.status_code == 200:
            st.session_state.rag_config = config_response.json()
    except:
        pass

    # HyDE 开关
    enable_hyde = st.checkbox(
        "启用 HyDE",
        value=st.session_state.rag_config.get("enable_hyde", True),
        help="生成假设文档辅助检索，可提升召回率（会增加延迟）"
    )

    # 查询扩展开关
    enable_query_expansion = st.checkbox(
        "启用查询扩展",
        value=st.session_state.rag_config.get("enable_query_expansion", True),
        help="生成同义问题增加召回率（会增加延迟）"
    )

    # 配置变更时同步到服务器
    if enable_hyde != st.session_state.rag_config.get("enable_hyde") or enable_query_expansion != st.session_state.rag_config.get("enable_query_expansion"):
        try:
            requests.post(
                f"{API_URL}/config",
                json={"enable_hyde": enable_hyde, "enable_query_expansion": enable_query_expansion},
                timeout=5
            )
            st.session_state.rag_config = {"enable_hyde": enable_hyde, "enable_query_expansion": enable_query_expansion}
        except Exception as e:
            st.warning(f"配置同步失败: {e}")

    # 清空对话
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

    # 系统信息
    st.divider()
    st.info("""
    **系统信息**
    - 版本：v1.0.0
    - 模型：Qwen2.5-7B / GPT-3.5
    - 向量库：Milvus
    - Embedding: BGE-M3
    """)

# ==================== 主界面 ====================
# 标题
st.markdown('<p class="main-header">🚗 汽车电子软件知识库助手</p>', unsafe_allow_html=True)
st.markdown("---")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # 如果是助手消息，显示思考过程
        if message["role"] == "assistant" and message.get("thinking"):
            with st.expander("💭 查看思考过程", expanded=False):
                st.markdown(message["thinking"])

        st.markdown(message["content"])

        # 如果是助手消息，显示来源
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 查看参考来源", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**来源 {i}:**")
                    st.markdown(f"> {source[:300]}..." if len(source) > 300 else f"> {source}")

# 聊天输入框
if prompt := st.chat_input("请输入您的问题，例如：ESP 初始化失败错误码是多少？"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 生成助手回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 思考中...")
        
        try:
            # 根据模式选择接口
            if query_mode == "Agent 模式":
                api_endpoint = f"{API_URL}/agent"
                spinner_text = "Agent 正在思考中..."
            else:
                api_endpoint = f"{API_URL}/query"
                spinner_text = "正在检索知识库并生成答案..."

            # 调用 API
            with st.spinner(spinner_text):
                response = requests.post(
                    api_endpoint,
                    json={"question": prompt},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    thinking = result.get("thinking", "")
                    sources = result.get("sources", [])

                    # 显示思考过程（可折叠）
                    if thinking:
                        with st.expander("💭 查看思考过程", expanded=False):
                            st.markdown(thinking)

                    # 显示答案
                    message_placeholder.markdown(answer)

                    # 显示来源
                    if sources:
                        with st.expander("📚 查看参考来源", expanded=True):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**来源 {i}:**")
                                st.markdown(f"> {source[:500]}..." if len(source) > 500 else f"> {source}")

                    # 保存到会话
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "thinking": thinking,
                        "sources": sources
                    })
                    
                    # 成功提示
                    st.success("✅ 回答完成")
                    
                else:
                    message_placeholder.markdown(f"❌ 请求失败：{response.status_code}")
                    st.error(f"错误信息：{response.text}")
                    
        except requests.exceptions.ConnectionError:
            message_placeholder.markdown("❌ 无法连接到 API 服务")
            st.error("请确保 API 服务正在运行：`python -m src.api`")
        except requests.exceptions.Timeout:
            message_placeholder.markdown("❌ 请求超时")
            st.error("问题可能比较复杂，请重试或联系管理员")
        except Exception as e:
            message_placeholder.markdown(f"❌ 发生错误：{str(e)}")
            st.error(f"详细错误：{str(e)}")

# ==================== 底部信息 ====================
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📖 使用提示**")
    st.markdown("- 问题尽量具体明确")
    st.markdown("- 可以追问获取更多信息")
    st.markdown("- 点击来源查看原文档")
    if query_mode == "Agent 模式":
        st.markdown("- Agent 模式可使用：知识库检索、网页搜索、计算器、Python代码")
with col2:
    st.markdown("**📊 系统统计**")
    st.markdown(f"- 对话轮数：{len(st.session_state.messages)//2}")
    st.markdown(f"- 最后更新：{datetime.now().strftime('%H:%M')}")
with col3:
    st.markdown("**🔧 技术支持**")
    st.markdown("- 问题反馈：联系 IT 部门")
    st.markdown("- 文档更新：联系知识库管理员")