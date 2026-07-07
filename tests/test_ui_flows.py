"""Test luồng UI thật của app bằng streamlit.testing.v1.AppTest (provider Demo).

Bộ test này chống hồi quy cho bug từng gặp: gửi chat khi memory tắt
làm trim mất lịch sử đã lưu của session.
"""
from pathlib import Path

import pytest
from langchain_core.chat_history import InMemoryChatMessageHistory
from streamlit.testing.v1 import AppTest

import streamlit_app as app

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def at_seeded():
    """AppTest đã seed sẵn 20 message (10 lượt) cho session mặc định demo-user.

    Cap mặc định của app: max_turns=8 -> max_messages=16 < 20,
    đủ để quan sát hành vi trim.
    """
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, f"app lỗi ngay khi boot: {at.exception}"
    hist = InMemoryChatMessageHistory()
    for i in range(10):
        hist.add_user_message(f"câu hỏi {i}")
        hist.add_ai_message(f"trả lời {i}")
    at.session_state["histories"] = {"demo-user": hist}
    return at


def _stored(at: AppTest):
    """Lấy danh sách message đã lưu của session demo-user."""
    return at.session_state["histories"]["demo-user"].messages


def test_memory_off_gui_tin_khong_dong_vao_lich_su(at_seeded):
    """Memory tắt: gửi tin KHÔNG được làm thay đổi lịch sử đã lưu (bug cũ: 20 -> 16)."""
    at_seeded.toggle[0].set_value(False)
    at_seeded.chat_input[0].set_value("lượt chat không memory")
    at_seeded.run()
    assert not at_seeded.exception
    assert len(_stored(at_seeded)) == 20


def test_memory_on_gui_tin_trim_ve_cap(at_seeded):
    """Memory bật: sliding window giữ đúng cap (16) sau khi gửi tin."""
    at_seeded.chat_input[0].set_value("lượt chat có memory")
    at_seeded.run()
    assert not at_seeded.exception
    assert len(_stored(at_seeded)) == 16


def test_rerun_memory_on_ap_cap_ngay(at_seeded):
    """Memory bật: chỉ cần rerun (đổi slider...) là cap được áp, không cần lượt chat mới."""
    at_seeded.run()
    assert not at_seeded.exception
    assert len(_stored(at_seeded)) == 16


def test_doi_provider_reset_model_ve_mac_dinh():
    """Đổi provider: ô Model reset về model mặc định của provider mới."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].set_value("OpenAI")
    at.run()
    assert not at.exception
    assert at.sidebar.text_input[0].value == app.default_model("OpenAI")


def test_model_giu_chinh_sua_qua_rerun_khong_lien_quan():
    """Sửa tay ô Model rồi rerun không liên quan (đổi temperature): giá trị được giữ."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    at.sidebar.selectbox[0].set_value("OpenAI")
    at.run()
    at.sidebar.text_input[0].set_value("gpt-tuy-chinh")
    at.run()
    at.sidebar.slider[0].set_value(0.5)
    at.run()
    assert not at.exception
    assert at.sidebar.text_input[0].value == "gpt-tuy-chinh"


def test_memory_off_hien_caption_giai_thich(at_seeded):
    """Memory tắt: UI phải có caption giải thích lượt chat sẽ không được lưu."""
    at_seeded.toggle[0].set_value(False)
    at_seeded.run()
    assert not at_seeded.exception
    texts = [str(c.value) for c in at_seeded.caption]
    assert any("Memory tắt" in t for t in texts)
