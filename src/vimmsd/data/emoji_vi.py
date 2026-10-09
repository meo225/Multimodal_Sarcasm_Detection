"""Nghĩa tiếng Việt của các emoji hay gặp, dùng thay cho tên tiếng Anh của `emoji.demojize`.

Gồm 100 emoji xuất hiện nhiều nhất trong caption tập train ViMMSD (chiếm khoảng 77% số lần xuất hiện
emoji) và vài emoji hay dùng để mỉa mai. Khóa là emoji đã bỏ variation selector (U+FE0F) và màu da;
emoji không có trong bảng vẫn được demojize sang tên tiếng Anh."""

EMOJI_VI = {
    # mặt cười, cảm xúc
    "🤣": "cười lăn lộn", "😂": "cười ra nước mắt", "😹": "mèo cười ra nước mắt",
    "😆": "cười híp mắt", "😀": "mặt cười toe", "😃": "cười tươi", "😁": "cười nhe răng",
    "😅": "cười toát mồ hôi", "🙂": "mặt cười mỉm", "🙃": "mặt cười lộn ngược", "☺": "mặt cười",
    "🥲": "cười trong nước mắt", "🥹": "mặt cố nín khóc", "😭": "khóc nức nở", "😢": "mặt khóc",
    "🥰": "mặt cười đầy tim", "😍": "mắt hình trái tim", "😘": "hôn gió", "😚": "hôn nhắm mắt",
    "🤩": "mắt hình ngôi sao", "😎": "mặt đeo kính râm", "🥳": "mặt ăn mừng", "😇": "mặt thiên thần",
    "😌": "mặt nhẹ nhõm", "😋": "mặt thèm ăn", "🤤": "chảy nước miếng", "😉": "nháy mắt",
    "😏": "cười nhếch mép", "🤭": "che miệng cười", "🤔": "mặt suy nghĩ", "🤨": "nhướn mày nghi ngờ",
    "😐": "mặt không cảm xúc", "😑": "mặt vô cảm", "😒": "mặt chán chường", "🙄": "đảo mắt",
    "😬": "mặt nhăn răng", "😕": "mặt bối rối", "🙁": "mặt hơi buồn", "😞": "mặt thất vọng",
    "🥺": "mặt năn nỉ", "🥴": "mặt choáng váng", "😳": "mặt đỏ bừng", "😮": "mặt há hốc",
    "😲": "mặt kinh ngạc", "😱": "hét lên sợ hãi", "😰": "mặt lo lắng toát mồ hôi",
    "🤯": "nổ tung đầu", "😤": "mặt hậm hực", "😠": "mặt tức giận", "😡": "mặt giận dữ",
    "🥶": "mặt lạnh cóng", "🥵": "mặt nóng bừng", "🤡": "mặt hề", "💀": "đầu lâu",
    "☠": "đầu lâu xương chéo", "👻": "con ma", "💩": "cục phân",
    # tay, cử chỉ
    "👍": "ngón cái giơ lên", "👌": "tay ok", "✌": "tay chữ V", "🤟": "tay yêu bạn",
    "🙏": "chắp tay", "🙌": "giơ hai tay", "🫶": "tay hình trái tim", "🫰": "bắn tim",
    "🤌": "chụm ngón tay", "👉": "tay chỉ sang phải", "👇": "tay chỉ xuống", "🫵": "chỉ vào bạn",
    "🤷": "nhún vai", "🤦": "tay ôm mặt", "💃": "cô gái nhảy múa", "👀": "đôi mắt", "👁": "con mắt",
    # trái tim
    "❤": "trái tim", "♥": "trái tim", "❣": "trái tim", "💓": "trái tim đập", "💕": "hai trái tim",
    "💙": "tim xanh", "🖤": "tim đen", "💔": "tim vỡ", "❤‍🔥": "trái tim rực lửa",
    # dấu, ký hiệu: giữ nghĩa dấu câu để PhoBERT đọc như dấu câu thường
    "‼": "!!", "❗": "!", "❓": "?", "✅": "dấu tích xanh", "❌": "dấu x", "⚠": "cảnh báo",
    "🚨": "đèn báo động", "📍": "ghim vị trí", "📌": "đinh ghim", "🔹": "hình thoi xanh",
    # đồ vật, thiên nhiên
    "🔥": "lửa", "✨": "lấp lánh", "⚡": "tia sét", "❄": "bông tuyết", "🌸": "hoa anh đào",
    "🎉": "pháo giấy", "🎇": "pháo hoa", "🎁": "hộp quà", "⏰": "đồng hồ báo thức", "📸": "máy ảnh",
    "🚗": "ô tô", "🥕": "cà rốt", "🐧": "chim cánh cụt", "🐊": "cá sấu", "🦈": "cá mập",
    "🇻🇳": "cờ Việt Nam",
}
