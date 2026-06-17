"""
session_manager.py — R8+R9: Quản lý lịch sử chat persistent + multi-session
Lưu mỗi session thành file JSON riêng trong ./chat_sessions/
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path("./chat_sessions")


class SessionManager:
    """Quản lý các phiên hội thoại, lưu/load từ disk."""

    def __init__(self, sessions_dir: str = None):
        self._dir = Path(sessions_dir) if sessions_dir else SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._current_id: Optional[str] = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_session(self, name: str = None) -> str:
        """Tạo session mới, trả về session_id."""
        sid = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        if not name:
            name = f"Phiên {now.strftime('%Y-%m-%d %H:%M')}"
        data = {
            "id": sid,
            "name": name,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "messages": [],
        }
        self._write(sid, data)
        self._current_id = sid
        logger.info(f"Tạo session mới: {sid} — {name}")
        return sid

    def get_session(self, sid: str) -> Optional[dict]:
        """Load session từ disk."""
        path = self._path(sid)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Không thể load session {sid}: {e}")
            return None

    def list_sessions(self) -> list[dict]:
        """Danh sách tất cả sessions, sắp xếp mới nhất trước."""
        sessions = []
        for p in self._dir.glob("*.json"):
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "id": data["id"],
                    "name": data.get("name", data["id"]),
                    "updated_at": data.get("updated_at", ""),
                    "msg_count": len(data.get("messages", [])),
                })
            except Exception:
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def add_message(self, sid: str, role: str, content: str):
        """Thêm tin nhắn vào session và lưu xuống disk."""
        data = self.get_session(sid)
        if not data:
            logger.warning(f"Session {sid} không tồn tại, tạo mới.")
            sid = self.create_session()
            data = self.get_session(sid)

        now = datetime.now(timezone.utc).isoformat()
        data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": now,
        })
        data["updated_at"] = now
        self._write(sid, data)

    def rename_session(self, sid: str, new_name: str):
        """Đổi tên session."""
        data = self.get_session(sid)
        if not data:
            return
        data["name"] = new_name
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(sid, data)

    def delete_session(self, sid: str):
        """Xóa session khỏi disk."""
        path = self._path(sid)
        if path.exists():
            path.unlink()
            logger.info(f"Đã xóa session: {sid}")

    def get_messages_as_gradio(self, sid: str) -> list[dict]:
        """Trả về messages dạng Gradio 6 dict format."""
        data = self.get_session(sid)
        if not data:
            return []
        return [
            {"role": m["role"], "content": m["content"]}
            for m in data.get("messages", [])
        ]

    # ------------------------------------------------------------------
    # Current session helpers
    # ------------------------------------------------------------------

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id

    def ensure_current(self) -> str:
        """Đảm bảo luôn có session hiện tại, tạo mới nếu chưa có."""
        if self._current_id and self.get_session(self._current_id):
            return self._current_id
        # Thử load session gần nhất
        sessions = self.list_sessions()
        if sessions:
            self._current_id = sessions[0]["id"]
            return self._current_id
        # Tạo mới
        return self.create_session()

    def switch_to(self, sid: str) -> list[dict]:
        """Chuyển sang session khác, trả về messages để hiển thị."""
        self._current_id = sid
        return self.get_messages_as_gradio(sid)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _path(self, sid: str) -> Path:
        return self._dir / f"{sid}.json"

    def _write(self, sid: str, data: dict):
        try:
            with open(self._path(sid), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Không thể lưu session {sid}: {e}")
