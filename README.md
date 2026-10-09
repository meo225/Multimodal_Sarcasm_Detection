# Môn học: Khai thác dữ liệu truyền thông xã hội

## Đồ án: Multimodal Sarcasm Detection trên mạng xã hội Việt Nam (ViMMSD)

### 1. Mô tả dự án

Đồ án xây dựng hệ thống phát hiện mỉa mai (sarcasm) trên các bài đăng mạng xã hội Việt Nam có kết hợp cả **ảnh và văn bản (caption)**. Nhiều bài đăng chỉ mỉa mai khi xét đồng thời cả hình ảnh lẫn câu chữ — ví dụ ảnh và text có nội dung mâu thuẫn nhau — nên bài toán đòi hỏi mô hình phải học được cách kết hợp thông tin từ hai modality (multimodal fusion) thay vì chỉ dựa vào text như các hệ thống phát hiện sarcasm truyền thống.

Dự án sử dụng bộ dữ liệu **ViMMSD (Vietnamese Multimodal Sarcasm Detection)**, được công bố trong khuôn khổ UIT Data Science Challenge 2024, gồm các cặp (ảnh, caption) thu thập từ mạng xã hội Việt Nam.

### 2. Bài toán (Problem Statement)

Given một cặp (ảnh, caption) từ một bài đăng mạng xã hội, phân loại bài đăng đó vào 1 trong 4 nhãn mỉa mai, dựa trên nguồn gốc của sự mỉa mai (đến từ text, từ ảnh, từ cả hai, hoặc không mỉa mai).

Thách thức chính của bài toán:
- Phải học được sự **mâu thuẫn/bổ trợ giữa 2 modality**, không thể giải quyết tốt chỉ bằng text hoặc chỉ bằng ảnh.
- Dữ liệu **mất cân bằng nghiêm trọng** giữa các lớp (một lớp chỉ có vài chục mẫu trong khi lớp khác có hàng nghìn mẫu).
- Văn bản tiếng Việt trên mạng xã hội có nhiều teencode, viết tắt, emoji, cần tiền xử lý riêng.

### 3. Input / Output

**Input**: một bài đăng mạng xã hội gồm:
- Một ảnh (image)
- Một đoạn caption/văn bản tiếng Việt đi kèm ảnh

**Output**: nhãn phân loại thuộc 1 trong 4 lớp:
- `not-sarcasm` — không mỉa mai
- `text-sarcasm` — mỉa mai thể hiện qua văn bản
- `image-sarcasm` — mỉa mai đến từ mâu thuẫn giữa ảnh và văn bản
- `multi-sarcasm` — cần kết hợp cả ảnh và văn bản mới nhận biết được sự mỉa mai

### 4. Các bước thực hiện (Step by step)

1. **Thu thập & khám phá dữ liệu (EDA)**
   - Tải bộ dữ liệu ViMMSD.
   - Phân tích phân bố nhãn, độ dài caption, đặc điểm ảnh để hiểu rõ mức độ mất cân bằng và tính chất dữ liệu.

2. **Tiền xử lý dữ liệu**
   - Văn bản: tách từ tiếng Việt, chuẩn hóa teencode/viết tắt, xử lý emoji.
   - Ảnh: resize, chuẩn hóa theo yêu cầu của encoder ảnh sử dụng.
   - Ảnh → text (chạy offline một lần cho toàn bộ ảnh, lưu cache):
     - **OCR** (PaddleOCR phát hiện vùng chữ, VietOCR nhận dạng): trích chữ xuất hiện trong ảnh. Nhiều mẫu `image-sarcasm`/`multi-sarcasm` là meme có chữ, nội dung mỉa mai nằm ngay trong chữ đó (xem `reports/dataset_report.md`).
     - **Mô tả ảnh bằng VLM** (Vintern-1B, mô hình đa phương thức tiếng Việt): sinh 1–2 câu mô tả nội dung ảnh (ai/cái gì, hành động, bối cảnh, biểu cảm). Prompt trung tính, không nhắc tới mỉa mai, để không đưa phán đoán nhãn vào input.
     - Chữ trong ảnh và mô tả ảnh được ghép thành segment thứ hai của PhoBERT, cạnh caption, cho cả mô hình chỉ dùng text lẫn mô hình fusion.

3. **Xây dựng baseline đơn modality**
   - Mô hình chỉ dùng text (fine-tune PhoBERT).
   - Mô hình chỉ dùng ảnh (CLIP/ViT image encoder).
   - Dùng để làm cơ sở so sánh, chứng minh giá trị của việc kết hợp đa modality.

4. **Xây dựng mô hình multimodal fusion**
   - Baseline fusion: kết hợp đặc trưng text (PhoBERT) và ảnh (CLIP) bằng phép nối (concat) qua một lớp MLP phân loại.
   - Nâng cấp: thử nghiệm cơ chế cross-attention giữa hai modality để mô hình học tốt hơn sự mâu thuẫn/bổ trợ giữa ảnh và text.

5. **Xử lý mất cân bằng dữ liệu**
   - Áp dụng các kỹ thuật như class weighting hoặc focal loss, đặc biệt cho lớp có rất ít mẫu.

6. **Thử nghiệm mở rộng (nếu còn thời gian)**
   - Ablation đóng góp của text trích từ ảnh: chỉ caption, + OCR, + mô tả VLM, + cả hai.
   - So sánh với cách tiếp cận zero-shot/few-shot bằng mô hình ngôn ngữ đa phương thức (LLM đa modal).

7. **Đánh giá mô hình**
   - Đánh giá bằng Precision, Recall, F1 (ưu tiên macro-F1 do dữ liệu mất cân bằng).
   - Phân tích ma trận nhầm lẫn (confusion matrix) và các trường hợp dự đoán sai để hiểu hạn chế của mô hình.

8. **Phân tích & tổng kết**
   - So sánh hiệu quả giữa các mô hình (unimodal vs fusion vs cross-attention).
   - Rút ra nhận xét về đóng góp thực sự của từng modality trong việc phát hiện mỉa mai.
   - Viết báo cáo và chuẩn bị trình bày kết quả.

### 5. Hướng phân tích chuyên sâu (mở rộng)

Ngoài pipeline chính ở trên, nhóm có thời gian nên đào sâu thêm các hướng phân tích sau để làm phong phú phần kết quả/thảo luận trong báo cáo, thay vì chỉ dừng ở việc báo cáo chỉ số F1:

1. **Kiểm tra shortcut learning / spurious correlation**
   - Nhiều dataset sarcasm multimodal (được ghi nhận trong literature, ví dụ MMSD2.0) bị model "ăn gian" — chỉ dựa vào vài từ khóa/emoji trong text để đoán đúng phần lớn mà không thực sự dùng thông tin ảnh.
   - Cách kiểm tra: lấy các mẫu `multi-sarcasm`/`image-sarcasm`, **tráo đổi ảnh** giữa các mẫu (giữ nguyên text), quan sát xem dự đoán của model có thay đổi tương ứng hay không.
   - Nếu dự đoán không đổi khi tráo ảnh → model đang không thực sự học multimodal reasoning, chỉ dựa vào text.

2. **Trực quan hóa khả năng diễn giải (interpretability)**
   - Vẽ attention map từ lớp cross-attention để thể hiện vùng ảnh mà model chú ý khi đưa ra dự đoán.
   - Highlight các từ trong caption được model chú ý nhiều nhất.
   - Dùng để minh họa trực quan trong báo cáo/demo, giúp giải thích quyết định của model thay vì chỉ đưa ra con số.

3. **Đào sâu bài toán mất cân bằng dữ liệu**
   - Lớp `text-sarcasm` có rất ít mẫu trong tập train, cần một nghiên cứu so sánh nhỏ thay vì chỉ áp dụng một kỹ thuật xử lý duy nhất.
   - So sánh hiệu quả của: class weighting, focal loss, oversampling, và data augmentation cho text (ví dụ back-translation) — đánh giá kỹ thuật nào thực sự cải thiện được khả năng nhận diện lớp hiếm này.

4. **So sánh với mô hình ngôn ngữ đa phương thức (LLM đa modal) theo hướng zero-shot/few-shot**
   - Đánh giá một mô hình đa modal có sẵn (ví dụ Gemini, Qwen-VL, hoặc mô hình tiếng Việt nếu có) bằng cách prompting trực tiếp, không cần huấn luyện.
   - So sánh hiệu quả giữa mô hình nhỏ được fine-tune riêng cho bài toán và LLM tổng quát, phân tích trường hợp nào LLM làm tốt hơn (ví dụ hiểu ngữ cảnh văn hóa/tiếng Việt) và trường hợp nào mô hình fine-tune riêng vẫn vượt trội.
