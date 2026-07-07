# LangChain Chat Lab — CyberSoft Gen AI 01, phần 3

Tài liệu học LangChain cơ bản: hai notebook lý thuyết và một app Streamlit thực hành gom các khái niệm lại thành lab tương tác.

## Nội dung repo

| Thành phần | Mô tả |
|---|---|
| `intro_to_langchain.ipynb` | ChatOpenAI, ChatPromptTemplate, chain cơ bản (LCEL) |
| `intro_to_groq.ipynb` | ChatGroq, MessagesPlaceholder, RunnableWithMessageHistory (memory theo session) |
| `streamlit_app.py` | App "LangChain Chat Lab" 4 tab: Chat Lab, Prompt Template, Memory Viewer, Notebook Map |
| `tests/` | Test pytest: logic chain (bare mode) + luồng UI thật (streamlit AppTest) |

Output đã chạy sẵn trong hai notebook là một phần của bài học — không cần (và không nên) chạy lại nếu không có mục đích rõ ràng.

## Yêu cầu

- Python 3.11

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `requirements.txt` được pin đúng phiên bản LangChain 0.1.x của khóa học.
> Không nâng cấp tùy tiện: API LangChain 0.3+/1.x không tương thích với code bài học.

## Cấu hình

Tạo file `.env` ở gốc repo (file này đã được gitignore — không commit key):

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
# Tùy chọn, chọn provider mặc định: demo | openai | groq (mặc định: demo)
LANGCHAIN_APP_PROVIDER=demo
```

Chưa có API key vẫn dùng được **provider Demo**: chạy offline, giả lập câu trả lời để minh họa cơ chế memory/MessagesPlaceholder.

## Chạy app

```powershell
streamlit run streamlit_app.py
```

## Chạy test

```powershell
python -m pytest tests
```

## Chạy notebook

Mở bằng VS Code (hoặc editor hỗ trợ Jupyter) và chọn kernel `.venv`. Lưu ý: chạy lại notebook cần API key thật và sẽ ghi đè output đã lưu của bài học.
