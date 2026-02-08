import os
import shutil
import requests
from tqdm import tqdm
import zipfile
from pathlib import Path
import time

OUT_ROOT = "data/imagenet_subset"
# Прямая ссылка на релиз через GitHub Releases API (обходит ограничения)
URL = "https://github.com/HaohanWang/ImageNet-Sketch/releases/download/v1.0/imagenet-sketch.zip"
EXPECTED_SIZE = 1_200_000_000  # ~1.2 ГБ


def download_file(url, dest, max_retries=3):
    """Надёжное скачивание с редиректами и проверкой размера"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/octet-stream"
    }

    for attempt in range(max_retries):
        try:
            print(f"⬇️ Попытка {attempt + 1}/{max_retries}...")
            response = requests.get(url, stream=True, headers=headers, timeout=300)
            response.raise_for_status()

            # GitHub часто возвращает редирект на CDN — обрабатываем его
            if response.url != url and 'amazonaws.com' in response.url:
                print(f"🔄 Обнаружен редирект на CDN: {response.url}")

            total = int(response.headers.get('content-length', 0))
            if total < 100_000_000:  # Меньше 100 МБ — явно ошибка
                print(f"⚠️ Подозрительно маленький размер файла: {total / 1024 / 1024:.2f} МБ")
                if attempt == max_retries - 1:
                    raise ValueError("Скачан файл слишком мал — возможно, ошибка доступа")
                time.sleep(5)
                continue

            with open(dest, 'wb') as f, tqdm(
                    total=total, unit='iB', unit_scale=True,
                    desc=f"📥 Скачивание {Path(dest).name}"
            ) as bar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    bar.update(size)

            # Проверка итогового размера
            actual_size = os.path.getsize(dest)
            print(f"✅ Скачано: {actual_size / 1024 / 1024:.2f} МБ")
            if actual_size < EXPECTED_SIZE * 0.9:
                print(f"⚠️ Файл меньше ожидаемого ({EXPECTED_SIZE / 1024 / 1024:.0f} МБ). Повторная попытка...")
                os.remove(dest)
                time.sleep(5)
                continue

            return True

        except Exception as e:
            print(f"❌ Ошибка при скачивании: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)

    raise RuntimeError("Не удалось скачать файл после нескольких попыток")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    zip_path = os.path.join(OUT_ROOT, "imagenet-sketch.zip")
    extract_path = os.path.join(OUT_ROOT, "raw")

    # 1️⃣ Скачиваем архив (если ещё не скачан или повреждён)
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < EXPECTED_SIZE * 0.9:
        print("⬇️ Скачиваем официальный архив ImageNet-Sketch (~1.2 ГБ)...")
        download_file(URL, zip_path)
    else:
        print(f"✅ Архив уже скачан: {zip_path} ({os.path.getsize(zip_path) / 1024 / 1024:.2f} МБ)")

    # 2️⃣ Распаковываем
    if not os.path.exists(extract_path):
        print("📦 Распаковка архива...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("✅ Распаковка завершена")
        except zipfile.BadZipFile:
            print("❌ Файл повреждён или не является ZIP-архивом!")
            print("💡 Решение: удалите файл и перезапустите скрипт:")
            print(f"   del {zip_path}")
            return

    # 3️⃣ Определяем структуру
    sketch_root = os.path.join(extract_path, "imagenet-sketch", "sketch")

    if not os.path.exists(sketch_root):
        # Альтернативная структура (иногда распаковывается иначе)
        sketch_root = os.path.join(extract_path, "sketch")
        if not os.path.exists(sketch_root):
            print(f"❌ Не найдена папка со скетчами. Содержимое архива:")
            for root, dirs, files in os.walk(extract_path):
                level = root.replace(extract_path, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}{os.path.basename(root)}/')
                subindent = ' ' * 2 * (level + 1)
                for d in dirs[:3]:
                    print(f'{subindent}{d}/')
                for f in files[:3]:
                    print(f'{subindent}{f}')
            return

    # 4️⃣ Копируем в формат ImageFolder
    print("🔄 Конвертация в формат ImageFolder (PyTorch)...")
    class_dirs = [d for d in os.listdir(sketch_root) if os.path.isdir(os.path.join(sketch_root, d))]

    total_images = 0
    for class_id in tqdm(class_dirs, desc="Обработка классов"):
        src_dir = os.path.join(sketch_root, class_id)
        dst_dir = os.path.join(OUT_ROOT, class_id)
        os.makedirs(dst_dir, exist_ok=True)

        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for img_name in images:
            shutil.copy2(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))
        total_images += len(images)

    # 5️⃣ Итог
    print("\n✅ Готово!")
    print(f"📁 Сохранено в: {OUT_ROOT}")
    print(f"📊 Классов: {len(class_dirs)}")
    print(f"🖼️ Изображений: {total_images}")
    print(f"\nПример использования в PyTorch:\n")
    print("from torchvision.datasets import ImageFolder")
    print(f"dataset = ImageFolder(root='{OUT_ROOT}')")


if __name__ == "__main__":
    import socket

    socket.setdefaulttimeout(300)

    # Требуемые зависимости
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("❌ Требуются зависимости: pip install requests tqdm")
        exit(1)

    main()