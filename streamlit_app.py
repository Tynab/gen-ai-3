"""LangChain Chat Lab — app Streamlit minh họa các khái niệm LangChain cơ bản.

App gom kiến thức từ hai notebook của bài học (``intro_to_langchain.ipynb``,
``intro_to_groq.ipynb``) thành một lab tương tác gồm 4 tab:

- **Chat Lab**: hội thoại có/không memory theo session
  (``RunnableWithMessageHistory`` + ``MessagesPlaceholder``).
- **Prompt Template**: ví dụ ``ChatPromptTemplate`` nhiều biến (bài dịch thuật).
- **Memory Viewer**: xem và export JSON lịch sử đã lưu của từng session.
- **Notebook Map**: liên kết kiến thức notebook với từng phần của app.

Provider "Demo" là mặc định, chạy offline không cần API key; OpenAI/Groq cần
key trong file ``.env``. Code chủ đích bám API LangChain 0.1.x của khóa học
(kể cả ``RunnableWithMessageHistory`` đã deprecated) — không modernize.
"""
from __future__ import annotations

import json
import os
import uuid
import warnings
from html import escape
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# App dạy đúng pattern RunnableWithMessageHistory của khóa học, nên tắt cảnh báo
# deprecated để log không nhiễu.
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings(
    "ignore",
    category=LangChainDeprecationWarning,
    message=".*RunnableWithMessageHistory.*",
)

load_dotenv()

APP_TITLE = "LangChain Chat Lab"
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI đang dạy LangChain bằng ví dụ ngắn gọn, dễ hiểu. "
    "Khi người dùng hỏi bằng tiếng Việt, hãy trả lời bằng tiếng Việt. "
    "Nếu có lịch sử hội thoại, hãy tận dụng nó để giữ mạch ngữ cảnh."
)
# Nguồn sự thật duy nhất cho ánh xạ provider -> biến môi trường chứa API key.
PROVIDER_ENV_KEYS = {"OpenAI": "OPENAI_API_KEY", "Groq": "GROQ_API_KEY"}
# Ánh xạ kiến thức từ notebook sang phần tương ứng trong app (tab Notebook Map).
NOTEBOOK_MAP = [
    {
        "notebook": "intro_to_langchain.ipynb",
        "concepts": "ChatOpenAI, ChatPromptTemplate, chain cơ bản.",
        "app_section": "Chat Lab và Prompt Template.",
    },
    {
        "notebook": "intro_to_groq.ipynb",
        "concepts": "ChatGroq, ChatPromptTemplate, MessagesPlaceholder, RunnableWithMessageHistory.",
        "app_section": "Chat có memory theo session_id.",
    },
]


def init_state() -> None:
    """Khởi tạo session state của Streamlit cho lần chạy đầu tiên."""
    if "histories" not in st.session_state:
        st.session_state.histories = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = "demo-user"


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """Lấy (tạo nếu chưa có) lịch sử hội thoại của một session.

    Đây là ``get_session_history`` mà RunnableWithMessageHistory sẽ gọi
    với ``config["configurable"]["session_id"]``.
    """
    histories: dict[str,
                    InMemoryChatMessageHistory] = st.session_state.histories
    if session_id not in histories:
        histories[session_id] = InMemoryChatMessageHistory()
    return histories[session_id]


def trim_history(session_id: str, max_messages: int) -> None:
    """Cắt lịch sử về tối đa ``max_messages`` message gần nhất (sliding window)."""
    history = get_history(session_id)
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]


def message_role(message: BaseMessage) -> str:
    """Quy đổi loại message LangChain sang role hiển thị của st.chat_message."""
    return "user" if isinstance(message, HumanMessage) else "assistant"


def message_text(content: Any) -> str:
    """Ép nội dung message về chuỗi thuần.

    Content có thể là str, hoặc list các block (dạng multimodal) tùy provider —
    giữ nhánh list để phòng response khác chuẩn từ API bên ngoài.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(
                    str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def render_css() -> None:
    """Bơm CSS tùy biến cho header, sidebar và các thẻ trong Memory Viewer.

    Màu dùng hàm CSS ``light-dark()``: Streamlit đặt ``color-scheme`` trên
    container gốc (.stApp) theo theme đang hoạt động và đổi nó NGAY khi
    người dùng chuyển theme (không rerun script), nên CSS tự khớp theme
    mà không cần tải lại trang. Mỗi thuộc tính khai báo giá trị light
    trước làm fallback cho trình duyệt chưa hỗ trợ light-dark().
    """
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: #f7f9fb;
            background: light-dark(#f7f9fb, #171e28);
            border-right: 1px solid #dce3ea;
            border-right: 1px solid light-dark(#dce3ea, #2b3542);
        }
        .lab-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            align-items: center;
            justify-content: space-between;
            gap: 0.85rem;
            border-bottom: 1px solid #dde5ed;
            border-bottom: 1px solid light-dark(#dde5ed, #2b3542);
            padding-bottom: 0.55rem;
            margin-bottom: 0.75rem;
        }
        .lab-title {
            color: #132033;
            color: light-dark(#132033, #e8eef5);
            font-size: 1.28rem;
            font-weight: 720;
            line-height: 1.2;
        }
        .lab-subtitle {
            color: #536273;
            color: light-dark(#536273, #9fb0c2);
            font-size: 0.84rem;
            line-height: 1.35;
            margin-top: 0.12rem;
        }
        .status-pill {
            border: 1px solid #cfd9e3;
            border: 1px solid light-dark(#cfd9e3, #3a4756);
            border-radius: 999px;
            color: #24465d;
            color: light-dark(#24465d, #cfe0f0);
            background: #ffffff;
            background: light-dark(#ffffff, #1c2530);
            font-size: 0.76rem;
            padding: 0.28rem 0.58rem;
            white-space: nowrap;
        }
        @media (max-width: 760px) {
            .block-container {
                padding-top: 2rem;
            }
            .lab-header {
                grid-template-columns: 1fr;
                gap: 0.45rem;
            }
            .status-pill {
                width: fit-content;
            }
        }
        .memory-row {
            border: 1px solid #dce4ec;
            border: 1px solid light-dark(#dce4ec, #2c3a4a);
            border-radius: 8px;
            padding: 0.7rem 0.8rem;
            margin-bottom: 0.55rem;
            background: #ffffff;
            background: light-dark(#ffffff, #1c2530);
        }
        .memory-role {
            color: #4b5d70;
            color: light-dark(#4b5d70, #92a5ba);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .memory-text {
            color: #1b2635;
            color: light-dark(#1b2635, #e4ecf4);
            margin-top: 0.25rem;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(provider: str, session_id: str, context_enabled: bool) -> None:
    """Vẽ header của app kèm pill trạng thái (provider, session, memory)."""
    context_label = "Memory bật" if context_enabled else "Memory tắt"
    st.markdown(
        f"""
        <div class="lab-header">
            <div>
                <div class="lab-title">{APP_TITLE}</div>
                <div class="lab-subtitle">Một app nhỏ để thử ChatPromptTemplate, MessagesPlaceholder và lịch sử hội thoại theo session.</div>
            </div>
            <div class="status-pill">{provider} · {session_id} · {context_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def provider_is_ready(provider: str) -> bool:
    """Provider dùng được chưa? Demo luôn sẵn sàng; OpenAI/Groq cần key trong .env."""
    env_key = PROVIDER_ENV_KEYS.get(provider)
    return env_key is None or bool(os.getenv(env_key))


def ensure_provider_ready(provider: str) -> bool:
    """Guard dùng chung cho các tab: báo lỗi UI nếu provider thiếu API key."""
    if provider_is_ready(provider):
        return True
    st.error("Provider hiện tại chưa có API key trong .env.")
    return False


def report_model_error(exc: Exception) -> None:
    """Hiển thị lỗi gọi model theo một định dạng thống nhất giữa các tab."""
    st.error(f"Không gọi được model: {exc}")


def default_provider() -> str:
    """Đọc provider mặc định từ LANGCHAIN_APP_PROVIDER; không hợp lệ thì về Demo."""
    env_provider = os.getenv("LANGCHAIN_APP_PROVIDER", "Demo").strip().lower()
    provider_map = {"demo": "Demo", "openai": "OpenAI", "groq": "Groq"}
    return provider_map.get(env_provider, "Demo")


def default_model(provider: str) -> str:
    """Model mặc định theo provider (ưu tiên biến môi trường nếu có)."""
    if provider == "OpenAI":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "Groq":
        return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return "demo-langchain"


def demo_reply(prompt_value: Any) -> AIMessage:
    """Giả lập câu trả lời của model cho provider Demo (offline, không cần key).

    Đếm số lượt user có trong prompt để minh họa việc lịch sử đã thật sự được
    truyền qua ``MessagesPlaceholder`` khi memory bật.
    """
    messages = prompt_value.to_messages()
    user_messages = [message_text(msg.content)
                     for msg in messages if isinstance(msg, HumanMessage)]
    latest = user_messages[-1] if user_messages else ""
    remembered = max(len(user_messages) - 1, 0)
    if remembered:
        content = (
            f"Demo mode: mình nhận câu mới là: \"{latest}\".\n\n"
            f"Trong prompt hiện tại có {remembered} lượt người dùng trước đó, nên đây là ví dụ memory đang được truyền qua `MessagesPlaceholder`."
        )
    else:
        content = (
            f"Demo mode: mình nhận câu này độc lập: \"{latest}\".\n\n"
            "Nếu bật memory và hỏi tiếp cùng `session_id`, app sẽ đưa lịch sử vào prompt cho lượt sau."
        )
    return AIMessage(content=content)


def build_model(provider: str, model_name: str, temperature: float):
    """Khởi tạo chat model theo provider; Demo trả về RunnableLambda giả lập."""
    if not provider_is_ready(provider):
        raise RuntimeError(
            f"Thiếu {PROVIDER_ENV_KEYS[provider]} trong file .env.")
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name, temperature=temperature)
    if provider == "Groq":
        return ChatGroq(model=model_name, temperature=temperature)
    return RunnableLambda(demo_reply)


def build_chain(
    provider: str,
    model_name: str,
    temperature: float,
    system_prompt: str,
    context_enabled: bool,
):
    """Ghép prompt + model thành chain, có/không bọc memory tùy toggle.

    Nhánh memory bật giữ NGUYÊN pattern trong snippet đáp án ở tab Notebook Map
    (system + MessagesPlaceholder("history") + human, bọc RunnableWithMessageHistory).
    Nhánh memory tắt là chain thuần, không đọc/ghi lịch sử.
    """
    model = build_model(provider, model_name, temperature)

    if context_enabled:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ]
        )
        chain = prompt | model

        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    return prompt | model


def invoke_chat(
    user_input: str,
    provider: str,
    model_name: str,
    temperature: float,
    system_prompt: str,
    context_enabled: bool,
    session_id: str,
    max_messages: int,
) -> str:
    """Chạy một lượt chat và trả về câu trả lời dạng text.

    Memory bật: truyền ``session_id`` qua config để RunnableWithMessageHistory
    tự đọc/ghi lịch sử, rồi trim ngay để store không vượt cap — việc cắt lịch sử
    thuộc về nơi ghi, caller không phải nhớ. Memory tắt: invoke chain thuần,
    không đụng vào lịch sử đã lưu.
    """
    chain = build_chain(provider, model_name, temperature,
                        system_prompt, context_enabled)

    if context_enabled:
        result = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        trim_history(session_id, max_messages)
    else:
        result = chain.invoke({"input": user_input})

    return message_text(result.content)


def invoke_template(
    provider: str,
    model_name: str,
    temperature: float,
    source_lang: str,
    target_lang: str,
    tone: str,
    text: str,
) -> str:
    """Chạy ví dụ ChatPromptTemplate nhiều biến: dịch ``text`` theo cấu hình.

    Đây là bài học riêng về template (không memory) nên tự dựng prompt,
    chỉ dùng chung build_model với phần chat.
    """
    model = build_model(provider, model_name, temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Bạn là một dịch giả chuyên nghiệp. "
                "Hãy dịch nội dung được cung cấp từ {source_lang} sang {target_lang} "
                "với giọng văn {tone}. Chỉ trả về bản dịch, không giải thích gì thêm.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | model

    result = chain.invoke(
        {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "tone": tone,
            "text": text,
        }
    )
    return message_text(result.content)


def render_sidebar() -> dict[str, Any]:
    """Vẽ sidebar cấu hình và trả về config dùng chung cho các tab."""
    st.sidebar.title("Cấu hình")

    provider_options = ["Demo", "OpenAI", "Groq"]
    provider = st.sidebar.selectbox(
        "Provider",
        provider_options,
        index=provider_options.index(default_provider()),
    )

    # Widget không key: đổi provider -> đổi value mặc định -> Streamlit tự tạo
    # widget mới (reset về default); các rerun khác giữ nguyên giá trị user sửa.
    model_name = st.sidebar.text_input(
        "Model",
        value=default_model(provider),
        disabled=provider == "Demo",
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    context_enabled = st.sidebar.toggle("Dùng memory theo session", value=True)
    max_turns = st.sidebar.slider(
        "Giữ tối đa bao nhiêu lượt gần nhất", 2, 20, 8)

    st.sidebar.divider()
    session_id = st.sidebar.text_input(
        "Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id.strip() or "demo-user"

    cols = st.sidebar.columns(2)
    if cols[0].button("Session mới", use_container_width=True):
        st.session_state.session_id = f"user-{uuid.uuid4().hex[:6]}"
        st.rerun()
    if cols[1].button("Xóa session", use_container_width=True):
        st.session_state.histories.pop(st.session_state.session_id, None)
        st.rerun()

    if st.sidebar.button("Xóa toàn bộ memory", use_container_width=True):
        st.session_state.histories = {}
        st.rerun()

    st.sidebar.divider()
    system_prompt = st.sidebar.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=150,
    )

    if not provider_is_ready(provider):
        st.sidebar.warning(
            f"Thiếu {PROVIDER_ENV_KEYS[provider]}. Chọn Demo hoặc thêm key vào .env.")

    return {
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "context_enabled": context_enabled,
        # Mỗi lượt chat gồm 1 message user + 1 message AI.
        "max_messages": max_turns * 2,
        "session_id": st.session_state.session_id,
        "system_prompt": system_prompt,
    }


def render_chat_tab(config: dict[str, Any]) -> None:
    """Tab Chat Lab: hội thoại với model, minh họa memory theo session."""
    history = get_history(config["session_id"])

    # Trim lúc render để cap mới (hạ slider) có hiệu lực ngay trên rerun;
    # memory tắt thì tuyệt đối không đụng vào lịch sử đã lưu.
    if config["context_enabled"]:
        trim_history(config["session_id"], config["max_messages"])
    else:
        st.caption(
            "Memory tắt: model không thấy lịch sử phía trên và lượt chat này sẽ không được lưu.")

    for message in history.messages:
        with st.chat_message(message_role(message)):
            st.markdown(message_text(message.content))

    user_input = st.chat_input("Nhập câu hỏi để thử memory LangChain...")
    if not user_input:
        return

    if not ensure_provider_ready(config["provider"]):
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.chat_message("assistant"):
            with st.spinner("LangChain đang chạy chain..."):
                answer = invoke_chat(
                    user_input=user_input,
                    provider=config["provider"],
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    system_prompt=config["system_prompt"],
                    context_enabled=config["context_enabled"],
                    session_id=config["session_id"],
                    max_messages=config["max_messages"],
                )
                st.markdown(answer)
    except Exception as exc:
        report_model_error(exc)


def render_template_tab(config: dict[str, Any]) -> None:
    """Tab Prompt Template: form dịch thuật chạy qua invoke_template."""
    with st.form("template-form"):
        col_a, col_b = st.columns(2)
        source_lang = col_a.text_input("Ngôn ngữ nguồn", value="Tiếng Việt")
        target_lang = col_b.text_input("Ngôn ngữ đích", value="English")
        tone = st.selectbox(
            "Giọng văn",
            ["Tự nhiên", "Chuyên nghiệp", "Thân thiện", "Ngắn gọn"],
        )
        text = st.text_area(
            "Nội dung cần dịch",
            value="LangChain giúp mình ghép prompt, model và memory thành một chain dễ quản lý.",
            height=140,
        )
        submitted = st.form_submit_button(
            "Chạy prompt template", use_container_width=True)

    if not submitted:
        return

    if not ensure_provider_ready(config["provider"]):
        return

    try:
        with st.spinner("Đang invoke prompt template..."):
            result = invoke_template(
                provider=config["provider"],
                model_name=config["model_name"],
                temperature=config["temperature"],
                source_lang=source_lang,
                target_lang=target_lang,
                tone=tone,
                text=text,
            )
        st.markdown("**Kết quả**")
        st.markdown(result)
    except Exception as exc:
        report_model_error(exc)


def render_memory_tab(config: dict[str, Any]) -> None:
    """Tab Memory Viewer: xem lịch sử từng session và tải về dạng JSON."""
    histories: dict[str,
                    InMemoryChatMessageHistory] = st.session_state.histories
    session_ids = sorted(histories.keys()) or [config["session_id"]]
    selected = st.selectbox(
        "Chọn session",
        session_ids,
        index=session_ids.index(
            config["session_id"]) if config["session_id"] in session_ids else 0,
    )
    history = get_history(selected)

    if not history.messages:
        st.info("Session này chưa có message nào.")
    else:
        for message in history.messages:
            role = "Human" if message_role(message) == "user" else "AI"
            safe_content = escape(message_text(message.content))
            st.markdown(
                f"""
                <div class="memory-row">
                    <div class="memory-role">{role}</div>
                    <div class="memory-text">{safe_content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    export_rows = [
        {"role": message_role(message),
         "content": message_text(message.content)}
        for message in history.messages
    ]
    st.download_button(
        "Tải memory JSON",
        data=json.dumps(export_rows, ensure_ascii=False, indent=2),
        file_name=f"{selected}-memory.json",
        mime="application/json",
        use_container_width=True,
        disabled=not export_rows,
    )


def render_notebook_tab() -> None:
    """Tab Notebook Map: đối chiếu kiến thức notebook và snippet đáp án chuẩn."""
    for item in NOTEBOOK_MAP:
        with st.expander(item["notebook"], expanded=True):
            st.markdown(f"**Kiến thức:** {item['concepts']}")
            st.markdown(f"**Được đưa vào app:** {item['app_section']}")

    # Snippet đáp án chuẩn của bài học — build_chain phải giữ đúng pattern này.
    st.code(
        """
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | model
chatbot = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)
        """.strip(),
        language="python",
    )


def main() -> None:
    """Điểm vào của app: dựng layout, sidebar và 4 tab nội dung."""
    st.set_page_config(page_title=APP_TITLE, page_icon="LC", layout="wide")
    init_state()
    render_css()
    config = render_sidebar()
    render_header(config["provider"], config["session_id"],
                  config["context_enabled"])

    chat_tab, template_tab, memory_tab, notebook_tab = st.tabs(
        ["Chat Lab", "Prompt Template", "Memory Viewer", "Notebook Map"]
    )
    with chat_tab:
        render_chat_tab(config)
    with template_tab:
        render_template_tab(config)
    with memory_tab:
        render_memory_tab(config)
    with notebook_tab:
        render_notebook_tab()


if __name__ == "__main__":
    main()
