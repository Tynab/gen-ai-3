# LangChain Chat Lab — CyberSoft Gen AI 01, phần 3

Tài liệu học LangChain cơ bản: hai notebook lý thuyết và một app Streamlit thực hành gom các khái niệm lại thành lab tương tác.

## Nội dung repo

| Thành phần | Mô tả |
|---|---|
| `intro_to_langchain.ipynb` | ChatOpenAI, ChatPromptTemplate, chain cơ bản (LCEL) |
| `intro_to_groq.ipynb` | ChatGroq, MessagesPlaceholder, RunnableWithMessageHistory (memory theo session) |
| `streamlit_app.py` | App "LangChain Chat Lab" 4 tab: Chat Lab, Prompt Template, Memory Viewer, Notebook Map |
| `tests/` | Test pytest: logic chain (bare mode) + luồng UI thật (streamlit AppTest) |

Output đã lưu sẵn trong hai notebook là một phần của bài học — không cần (và không nên) chạy lại nếu không có mục đích rõ ràng.

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

Tạo file `.env` ở gốc repo (đã gitignore — không commit key). Xem `.env.example` để biết cần những gì:

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5-mini
GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-20b
# Tùy chọn, chọn provider mặc định: demo | openai | groq (mặc định: demo)
LANGCHAIN_APP_PROVIDER=demo
```

**Tên model chỉ đọc từ `.env`, code không hardcode và không có giá trị fallback.** Thiếu `OPENAI_MODEL` thì app báo thẳng "Thiếu OPENAI_MODEL trong file .env." thay vì lặng lẽ gọi một model bạn không chọn.

Chưa có API key vẫn dùng được **provider Demo**: chạy offline, giả lập câu trả lời để minh họa cơ chế memory/MessagesPlaceholder.

### Lưu ý: model họ gpt-5 / o-series và `temperature`

`gpt-5*`, `o1*`, `o3*`, `o4*` **chỉ chấp nhận `temperature = 1`** — gửi giá trị khác thì OpenAI trả về:

```
400 Unsupported value: 'temperature' does not support 0.2 with this model.
```

Bỏ trống tham số cũng không cứu được, vì `langchain-openai 0.0.2` luôn tự gửi kèm `temperature=0.7`. App tự xử lý bằng `effective_temperature()`: gặp model bị khóa thì ép về 1 và **khóa luôn slider Temperature** để giao diện hiển thị đúng giá trị đang chạy. Trong notebook thì phải tự truyền `temperature=1`.

## Chạy app

```powershell
streamlit run streamlit_app.py
```

| Provider | Cần key | Model |
|---|---|---|
| Demo (mặc định) | không | offline, giả lập — dùng để học cơ chế memory |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` |
| Groq | `GROQ_API_KEY` | `GROQ_MODEL` |

## Chạy test

```powershell
python -m pytest tests
```

Chạy từ gốc repo (`tests/conftest.py` là thứ đưa root vào `sys.path`). Một test lẻ:

```powershell
python -m pytest tests/test_chains.py::test_gpt5_bi_ep_temperature_1
```

Toàn bộ test dùng **provider Demo**, nên không cần API key và không gọi mạng. Mặt trái: test không thể phát hiện lỗi phía API thật (bug `temperature` ở trên từng lọt lưới đúng vì lý do này). Sau khi sửa những chỗ liên quan tới `build_model` hoặc notebook, hãy chạy thật để kiểm chứng.

## Chạy notebook

Mở bằng VS Code (hoặc editor hỗ trợ Jupyter) và chọn kernel `.venv`. Chạy lại notebook cần API key thật và sẽ ghi đè output đã lưu của bài học.
