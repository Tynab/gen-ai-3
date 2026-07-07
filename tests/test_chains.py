"""Test chức năng lõi của streamlit_app ở chế độ bare (chạy python thuần).

Toàn bộ test dùng provider Demo nên không cần API key hay kết nối mạng.
"""
from types import SimpleNamespace

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory

import streamlit_app as app


@pytest.fixture
def bare_state(monkeypatch):
    """Thay st.session_state bằng namespace đơn giản vì bare mode không có runtime."""
    ns = SimpleNamespace(histories={})
    monkeypatch.setattr(app.st, "session_state", ns, raising=False)
    return ns


def _chat(text: str, *, context_enabled: bool, session_id: str,
          max_messages: int = 16) -> str:
    """Gọi invoke_chat với cấu hình Demo mặc định."""
    return app.invoke_chat(
        user_input=text,
        provider="Demo",
        model_name="demo-langchain",
        temperature=0.2,
        system_prompt="Bạn là trợ lý.",
        context_enabled=context_enabled,
        session_id=session_id,
        max_messages=max_messages,
    )


def test_memory_on_luu_va_truyen_lich_su(bare_state):
    """Memory bật: lượt 1 lưu đủ Human+AI; lượt 2 thấy lịch sử qua MessagesPlaceholder."""
    answer = _chat("Xin chào", context_enabled=True, session_id="s1")
    assert "Xin chào" in answer
    assert len(bare_state.histories["s1"].messages) == 2

    answer_2 = _chat("Tôi vừa nói gì?", context_enabled=True, session_id="s1")
    # demo_reply đếm số lượt user trước đó xuất hiện trong prompt
    assert "1 lượt" in answer_2


def test_memory_off_khong_crash_khong_luu(bare_state):
    """Memory tắt: chain thuần chạy được (không UnboundLocalError) và không lưu gì."""
    answer = _chat("Câu hỏi độc lập", context_enabled=False, session_id="s2")
    assert "Câu hỏi độc lập" in answer
    assert "s2" not in bare_state.histories or not bare_state.histories["s2"].messages


def test_memory_on_ap_cap_ngay_trong_invoke_chat(bare_state):
    """invoke_chat sở hữu việc trim: store không vượt cap ngay sau lượt memory bật."""
    hist = InMemoryChatMessageHistory()
    for i in range(10):
        hist.add_user_message(f"câu hỏi {i}")
        hist.add_ai_message(f"trả lời {i}")
    bare_state.histories["s3"] = hist

    _chat("lượt mới", context_enabled=True, session_id="s3")
    # 20 message cũ + 2 message mới, cap 16 -> phải còn đúng 16
    assert len(bare_state.histories["s3"].messages) == 16


def test_template_tra_ve_ket_qua(bare_state):
    """invoke_template trả về text chứa nội dung nguồn (Demo echo lại câu hỏi)."""
    result = app.invoke_template(
        provider="Demo",
        model_name="demo-langchain",
        temperature=0.2,
        source_lang="Tiếng Việt",
        target_lang="English",
        tone="Tự nhiên",
        text="LangChain rất thú vị.",
    )
    assert "LangChain rất thú vị." in result


def test_build_chain_memory_off_chay_duoc(bare_state):
    """Nhánh memory-off của build_chain trả về chain dùng được ngay."""
    chain = app.build_chain(
        provider="Demo",
        model_name="demo-langchain",
        temperature=0.2,
        system_prompt="Bạn là trợ lý.",
        context_enabled=False,
    )
    result = chain.invoke({"input": "ping"})
    assert "ping" in app.message_text(result.content)
