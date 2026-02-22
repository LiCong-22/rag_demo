# test_query.py
import requests

def test_rag(question):
    response = requests.post(
        "http://localhost:8000/query",
        json={"question": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n" + "🟢" * 30)
        print(f"❓ 问题：{question}")
        print("🟢" * 30)
        print(f"\n🤖 回答：\n{data['answer']}\n")
        
        print("📚 参考来源：")
        for i, src in enumerate(data["sources"], 1):
            print(f"\n  [{i}] {src[:150]}...")
        print("\n" + "🟢" * 30)
    else:
        print(f"❌ 请求失败：{response.status_code}")

# 测试问题
questions = [
    "ESP 初始化失败错误码是多少？",
    "CAN 总线默认波特率是多少？",
    "ESP 模块上电后做什么检查？"
]

for q in questions:
    test_rag(q)