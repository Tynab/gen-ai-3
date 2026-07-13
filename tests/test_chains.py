"""Test chức năng lõi của streamlit_app ở chế độ bare (chạy python thuần).

Toàn bộ test dùng provider Demo nên không cần API key hay kết nối mạng.
"""
from pathlib import Path
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


def test_default_model_lay_thang_tu_env(monkeypatch):
    """Tên model phải đọc từ .env, không hardcode trong code."""
    monkeypatch.setenv("OPENAI_MODEL", "openai-tu-env")
    monkeypatch.setenv("GROQ_MODEL", "groq-tu-env")

    assert app.default_model("OpenAI") == "openai-tu-env"
    assert app.default_model("Groq") == "groq-tu-env"


def test_default_model_thieu_env_thi_rong_chu_khong_fallback(monkeypatch):
    """Thiếu biến môi trường: trả rỗng, tuyệt đối không rơi về model hardcode."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    assert app.default_model("OpenAI") == ""
    assert app.default_model("Groq") == ""


def test_build_model_thieu_ten_model_thi_bao_loi_ro_rang(monkeypatch):
    """Không có model: báo lỗi nêu đúng tên biến .env, không gọi API mù."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    with pytest.raises(RuntimeError, match="OPENAI_MODEL"):
        app.build_model("OpenAI", "   ", 0.2)
    with pytest.raises(RuntimeError, match="GROQ_MODEL"):
        app.build_model("Groq", "", 0.2)


@pytest.mark.parametrize(
    "model_name, nhan_temperature",
    [
        ("gpt-5", False),
        ("gpt-5-mini", False),
        ("o1-mini", False),
        ("o3", False),
        ("openai/o1", False),  # Groq phục vụ model OpenAI dưới dạng có namespace
        ("gpt-4o-mini", True),
        ("openai/gpt-oss-20b", True),
        ("gpt-5x-cua-ai-do", True),  # khớp theo họ, không dính oan vì trùng prefix
    ],
)
def test_supports_temperature_khop_theo_ho_model(model_name, nhan_temperature):
    """Họ gpt-5/o-series chỉ chấp nhận temperature = 1; các model khác thì tự do."""
    assert app.supports_temperature(model_name) is nhan_temperature


def test_gpt5_bi_ep_temperature_1(monkeypatch):
    """Regression: gpt-5 + temperature != 1 -> API trả 400 'Unsupported value'.

    Bỏ hẳn tham số cũng không cứu được vì langchain-openai 0.0.2 luôn gửi kèm
    temperature mặc định 0.7 của nó, nên build_model phải ép đúng 1.0.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = app.build_model("OpenAI", "gpt-5-mini", 0.2)
    assert model.temperature == 1.0


def test_groq_cung_bi_ep_temperature_khi_model_bi_khoa(monkeypatch):
    """Luật temperature là thuộc tính của MODEL, không phải của nhánh provider."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    model = app.build_model("Groq", "openai/o1", 0.2)
    assert model.temperature == 1.0


def test_model_thuong_giu_nguyen_temperature_nguoi_dung_chon(monkeypatch):
    """Model không thuộc họ trên: slider temperature phải được tôn trọng."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = app.build_model("OpenAI", "gpt-4o-mini", 0.2)
    assert model.temperature == 0.2


def test_notebook_map_tro_dung_file_co_that():
    """Tab Notebook Map tồn tại để trỏ học viên tới notebook — tên file phải có thật."""
    repo_root = Path(app.__file__).resolve().parent

    for item in app.NOTEBOOK_MAP:
        assert (repo_root / item["notebook"]).is_file(), f"không có {item['notebook']}"
