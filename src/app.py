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
    
    # 检索参数
    st.subheader("🔍 检索配置")
    top_k = st.slider("检索文档数量", 1, 10, 3)
    
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
            # 调用 API
            with st.spinner("正在检索知识库并生成答案..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt},
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    
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
with col2:
    st.markdown("**📊 系统统计**")
    st.markdown(f"- 对话轮数：{len(st.session_state.messages)//2}")
    st.markdown(f"- 最后更新：{datetime.now().strftime('%H:%M')}")
with col3:
    st.markdown("**🔧 技术支持**")
    st.markdown("- 问题反馈：联系 IT 部门")
    st.markdown("- 文档更新：联系知识库管理员")