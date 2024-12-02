# @title Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')



# @title Google Drive에서 파일 복사

# 파일 복사 함수
def copy_if_exists(source_path, destination_path):
    import shutil
    import os
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
        print(f"Copied: {source_path} to {destination_path}")
    else:
        print(f"File not found: {source_path}")


# Colab 내 저장 경로
tar_path                = "/content/sketch.tar"
best_model_path         = "/content/best_model.pth"
checkpoint_path         = "/content/sketch_classification_checkpoint.pth"

# Google Drive 파일 경로
remote_tar_path         = "/content/drive/MyDrive/colab/toy project sketch classification/sketch.tar"
remote_best_model_path  = "/content/drive/MyDrive/colab/toy project sketch classification/best_model.pth"
remote_checkpoint_path  = "/content/drive/MyDrive/colab/toy project sketch classification/sketch_classification_checkpoint.pth"

# 파일 복사
copy_if_exists(remote_tar_path, tar_path)
# copy_if_exists(remote_best_model_path, best_model_path)
copy_if_exists(remote_checkpoint_path, checkpoint_path)



# @title 파일 압축 풀기
def untar():
	import tarfile

	# 파일 경로 및 대상 디렉토리 설정
	extract_dir = "/content/dataset"

	# 디렉토리 생성 (이미 존재하면 무시)
	os.makedirs(extract_dir, exist_ok=True)

	# tar 파일 압축 풀기
	with tarfile.open(tar_path, "r") as tar:
		tar.extractall(path=extract_dir)

	print(f"압축 해제 완료: {extract_dir}")



# @title 데이터셋 분리 (미리 수행해서 코드만 남김)
def split_dataset():
    pass
	# import os
	# import random
	# import shutil

	# # 데이터 경로 설정
	# base_dir = "/content/images/tx_000000000000"
	# train_dir = "/content/dataset/train"
	# val_dir = "/content/dataset/val"
	# test_dir = "/content/dataset/test"

	# # 데이터셋 디렉토리 생성
	# os.makedirs(train_dir, exist_ok=True)
	# os.makedirs(val_dir, exist_ok=True)
	# os.makedirs(test_dir, exist_ok=True)

	# # 각 클래스 처리
	# for class_name in os.listdir(base_dir):
	#     class_path = os.path.join(base_dir, class_name)

	#     if os.path.isdir(class_path):
	#         # 이미지 파일 목록 가져오기
	#         images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith(".png")]

	#         # 파일 무작위 셔플
	#         random.shuffle(images)

	#         # 파일 개수 계산
	#         total_count = len(images)
	#         train_count = int(total_count * 0.9)
	#         val_count = int(total_count * 0.05)

	#         # 데이터 분할
	#         train_files = images[:train_count]
	#         val_files = images[train_count:train_count + val_count]
	#         test_files = images[train_count + val_count:]

	#         # 파일 복사
	#         for file in train_files:
	#             class_train_dir = os.path.join(train_dir, class_name)
	#             os.makedirs(class_train_dir, exist_ok=True)
	#             shutil.copy(file, class_train_dir)

	#         for file in val_files:
	#             class_val_dir = os.path.join(val_dir, class_name)
	#             os.makedirs(class_val_dir, exist_ok=True)
	#             shutil.copy(file, class_val_dir)

	#         for file in test_files:
	#             class_test_dir = os.path.join(test_dir, class_name)
	#             os.makedirs(class_test_dir, exist_ok=True)
	#             shutil.copy(file, class_test_dir)

	# print("데이터 분할 완료!")
	# print(f"Training data saved in: {train_dir}")
	# print(f"Validation data saved in: {val_dir}")
	# print(f"Test data saved in: {test_dir}")



# @title 모델 학습, 검증, 저장, 테스트, 시각화

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt

# CLIP 기반 분류 모델 정의
class SketchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SketchClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip_model.parameters():
            param.requires_grad = False  # CLIP 이미지 인코더 Freeze
        self.classifier = nn.Linear(512, num_classes)  # Linear Classifier 추가

    def forward(self, images):
        image_features = self.clip_model.get_image_features(pixel_values=images)
        logits = self.classifier(image_features)
        return logits

# 데이터셋 경로 설정
train_dir = "/content/dataset/train"
val_dir = "/content/dataset/val"
test_dir = "/content/dataset/test"

# CLIP 전처리기
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 데이터 로더 생성
def create_dataloader(data_dir, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(data_dir, transform=lambda img: processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset

train_loader, train_dataset = create_dataloader(train_dir, batch_size=2048, shuffle=True)
val_loader, val_dataset = create_dataloader(val_dir, batch_size=2048, shuffle=False)
test_loader, test_dataset = create_dataloader(test_dir, batch_size=2048, shuffle=False)

# 모델 초기화
num_classes = len(train_dataset.classes)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SketchClassifier(num_classes).to(device)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

# 학습 함수
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": correct / len(dataloader.dataset)})
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

# 평가 함수
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            progress_bar.set_postfix({"Loss": loss.item(), "Accuracy": correct / len(dataloader.dataset)})
    accuracy = correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

# 체크포인트 저장 함수
def save_checkpoint(model, optimizer, epoch, train_history, val_history, best_val_acc, checkpoint_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_history": train_history,
        "val_history": val_history,
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with Validation Accuracy: {best_val_acc:.4f}")

# 체크포인트 불러오기 함수
def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        train_history = checkpoint["train_history"]
        val_history = checkpoint["val_history"]
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Checkpoint loaded from epoch {epoch} with Validation Accuracy: {best_val_acc:.4f}")
        return epoch, train_history, val_history, best_val_acc
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, [], [], 0.0

# 학습 설정
num_epochs = 10

# 초기화 또는 체크포인트 불러오기
start_epoch, train_history, val_history, best_val_acc = load_checkpoint(checkpoint_path, model, optimizer)

# 학습 루프
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    # 기록 업데이트
    train_history.append((train_loss, train_acc))
    val_history.append((val_loss, val_acc))

    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 최고 성능 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"  Best model saved with Validation Accuracy: {best_val_acc:.4f}")

    # 체크포인트 저장
    save_checkpoint(model, optimizer, epoch+1, train_history, val_history, best_val_acc, checkpoint_path)

# 테스트 평가
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")

# 기록 시각화
epochs = range(1, len(train_history) + 1)
train_losses, train_accuracies = zip(*train_history)
val_losses, val_accuracies = zip(*val_history)

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()
plt.show()



# @title 결과 파일 복사
copy_if_exists(best_model_path, remote_best_model_path)
copy_if_exists(checkpoint_path, remote_checkpoint_path)