import os
import random
import shutil
from src.config import VAL_DIR, DATASET_ROOT, SEED

TEST_RATIO = 0.20  # 20% of valid images per class will go to test
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def main():
    random.seed(SEED)

    test_dir = os.path.join(DATASET_ROOT, "test")
    os.makedirs(test_dir, exist_ok=True)

    class_names = [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]
    print("Classes in valid:", len(class_names))

    total_moved = 0
    for cls in class_names:
        src_cls = os.path.join(VAL_DIR, cls)
        dst_cls = os.path.join(test_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)

        imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(IMG_EXTS)]
        random.shuffle(imgs)

        k = max(1, int(len(imgs) * TEST_RATIO))
        chosen = imgs[:k]

        for f in chosen:
            shutil.move(os.path.join(src_cls, f), os.path.join(dst_cls, f))
            total_moved += 1

        print(f"{cls}: moved {len(chosen)}")

    print("\nDone.")
    print("Total moved:", total_moved)
    print("Test folder:", test_dir)

if __name__ == "__main__":
    main()