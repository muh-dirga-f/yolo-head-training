from src.model_splitter import YOLOSplitter
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_menu():
    print("\n=== YOLO Model Manager ===")
    print("1. Split and Save Model")
    print("2. Load Split Models")
    print("3. Run Prediction")
    print("4. Unload Models")
    print("5. Exit")
    print("========================")

def run_interactive():
    splitter = YOLOSplitter()

    while True:
        print_menu()
        choice = input("Pilih operasi (1-5): ")

        try:
            if choice == '1':
                splitter.split_model()
                splitter.save_split_models()
                logger.info("Model berhasil dipecah dan disimpan")

            elif choice == '2':
                splitter.load_split_models(load_dir="split_models")

            elif choice == '3':
                if not all([splitter.backbone, splitter.neck, splitter.head]):
                    logger.warning("Model belum dimuat! Silakan muat model terlebih dahulu (opsi 2)")
                    continue

                image_path = input("Masukkan path gambar (default: test.png): ").strip() or "test.png"
                logger.info(f"Running prediction on {image_path}...")

                splitter.predict(
                    source=image_path,
                    conf=0.25,
                    imgsz=640,
                    save=True,
                    project="output"
                )

                logger.info(f"Prediction complete. Results saved in 'output' directory")

            elif choice == '4':
                splitter.unload_models()
                logger.info("Model berhasil dibongkar dari memori")

            elif choice == '5':
                logger.info("Program selesai")
                break

            else:
                print("Pilihan tidak valid!")

        except Exception as e:
            logger.error(f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    run_interactive()
