import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ---------------------------------------------------------
# 1. HỆ THỐNG CẤU HÌNH TỰ ĐỘNG (AUTO-CONFIG SYSTEM)
# ---------------------------------------------------------
# Hàm này cực kỳ quan trọng: Nó tự động phát hiện môi trường đang chạy
def detect_environment():
    """Kiểm tra xem code đang chạy trên Kaggle hay Local (Laptop)"""
    if os.path.exists('/kaggle/input'):
        print("🌍 Đã phát hiện môi trường KAGGLE (Có thể dùng GPU & Model lớn)")
        return "kaggle"
    else:
        print("💻 Đã phát hiện môi trường LOCAL (Chỉ nên dùng CPU & Model nhỏ để test logic)")
        return "local"

ENV = detect_environment()

# Thiết lập các siêu tham số (Hyperparameters) dựa trên môi trường
if ENV == "kaggle":
    # --- CẤU HÌNH CHO KAGGLE (THỰC TẾ) ---
    MODEL_NAME = "distilbert-base-uncased"  # Model lớn hơn, mạnh hơn (có thể đổi thành bert-base)
    BATCH_SIZE = 16                         # Batch size lớn tận dụng GPU
    EPOCHS = 3
    SAMPLE_SIZE = None                      # Dùng toàn bộ dữ liệu (None)
else:
    # --- CẤU HÌNH CHO LOCAL (TEST LOGIC) ---
    MODEL_NAME = "prajjwal1/bert-tiny"      # Model siêu nhỏ (chỉ vài MB) để test logic cực nhanh
    BATCH_SIZE = 2                          # Batch size nhỏ để không tràn RAM CPU
    EPOCHS = 1                              # Chạy 1 epoch cho nhanh
    SAMPLE_SIZE = 20                        # Chỉ lấy 20 dòng dữ liệu để test

# Tự động chọn thiết bị (GPU nếu có, không thì CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Đang sử dụng thiết bị tính toán: {DEVICE}")
print(f"🧠 Đang sử dụng mô hình: {MODEL_NAME}")


# ---------------------------------------------------------
# 2. CHUẨN BỊ DỮ LIỆU (DATA PREPARATION)
# ---------------------------------------------------------
def load_and_prepare_data(env, sample_size):
    """
    Tạo dữ liệu giả lập (Dummy Data) cho mục đích demo.
    Trong thực tế, bạn sẽ đọc từ file CSV (ví dụ: pd.read_csv('fake_news.csv'))
    """
    print("\n--- BƯỚC 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU ---")
    
    # Dữ liệu giả lập (0: Tin thật, 1: Tin giả)
    data = {
        'text': [
            "Các nhà khoa học vừa phát hiện ra một hành tinh mới có sự sống.",
            "Ăn tỏi mỗi ngày sẽ giúp bạn miễn dịch hoàn toàn với mọi loại virus.",
            "Chính phủ công bố gói hỗ trợ kinh tế mới cho người dân.",
            "Người ngoài hành tinh đã hạ cánh xuống Trái Đất đêm qua và đang trà trộn vào đám đông.",
            "Thị trường chứng khoán hôm nay có dấu hiệu phục hồi nhẹ.",
            "Uống nước chanh nóng có thể chữa khỏi bệnh ung thư giai đoạn cuối."
        ] * 100, # Nhân bản dữ liệu lên cho nhiều
        'label': [0, 1, 0, 1, 0, 1] * 100
    }
    
    df = pd.DataFrame(data)
    
    # Nếu đang ở Local, chỉ lấy một lượng nhỏ dữ liệu để test logic (ví dụ: 20 dòng)
    if sample_size is not None:
        df = df.head(sample_size)
        print(f"📉 Đang ở Local: Cắt giảm dữ liệu xuống còn {sample_size} mẫu để test nhanh.")
    else:
        print(f"📈 Đang ở Kaggle: Sử dụng toàn bộ {len(df)} mẫu dữ liệu.")
        
    return Dataset.from_pandas(df)

# Khởi tạo Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """Chuyển đổi văn bản thành các con số (tensors) mà model hiểu được"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# Thực thi chuẩn bị dữ liệu
raw_dataset = load_and_prepare_data(ENV, SAMPLE_SIZE)

print("\n--- BƯỚC 2: TOKENIZE DỮ LIỆU ---")
print("Đang chuyển đổi văn bản thành Token...")
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# Chia dữ liệu thành tập Train (80%) và Test (20%)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
print("Đã chia dữ liệu thành tập Train và Test.")


# ---------------------------------------------------------
# 3. KHỞI TẠO MÔ HÌNH VÀ HUẤN LUYỆN (MODEL & TRAINING)
# ---------------------------------------------------------
print("\n--- BƯỚC 3: KHỞI TẠO MÔ HÌNH ---")
# Load model phân loại chuỗi (Sequence Classification) với 2 nhãn (0 và 1)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE) # Đẩy model vào CPU hoặc GPU

print("\n--- BƯỚC 4: BẮT ĐẦU HUẤN LUYỆN ---")
# Cấu hình các tham số huấn luyện (Tự động thay đổi theo môi trường)
training_args = TrainingArguments(
    output_dir="./results",          # Thư mục lưu kết quả
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    evaluation_strategy="epoch",     # Đánh giá model sau mỗi epoch
    logging_dir='./logs',            # Thư mục lưu log
    logging_steps=10,
    # Chỉ dùng CPU nếu đang ở local để tránh lỗi nếu máy không cài CUDA
    use_cpu=(DEVICE.type == 'cpu') 
)

# Khởi tạo Trainer của Hugging Face (Công cụ giúp train model dễ dàng)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Bắt đầu quá trình huấn luyện
print("🚀 Bắt đầu quá trình Training...")
trainer.train()

# ---------------------------------------------------------
# 4. ĐÁNH GIÁ VÀ LƯU KẾT QUẢ (EVALUATION & SAVING)
# ---------------------------------------------------------
print("\n--- BƯỚC 5: ĐÁNH GIÁ MÔ HÌNH ---")
eval_results = trainer.evaluate()
print(f"📊 Kết quả đánh giá: {eval_results}")

print("\n--- BƯỚC 6: LƯU MÔ HÌNH ---")
# Lưu model để sau này dùng lại mà không cần train lại
model_path = "./fake_news_model_saved"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)
print(f"✅ Đã lưu mô hình tại thư mục: {model_path}")
print("🎉 QUÁ TRÌNH HOÀN TẤT!")