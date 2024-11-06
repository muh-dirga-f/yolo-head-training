from src.model_splitter import YOLOSplitter
import logging
import os

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
        choice = input("Pilih operasi (1-5): ").strip()

        try:
            if choice == '1':
                model_path = input("Masukkan path model (kosongkan untuk pretrained): ").strip() or None
                splitter.split_model(model_path)
                save_dir = input("Masukkan direktori penyimpanan (default: split_models): ").strip() or "split_models"
                splitter.save_split_models(save_dir)
                logger.info("Model berhasil dipecah dan disimpan")

            elif choice == '2':
                load_dir = input("Masukkan direktori model (default: split_models): ").strip() or "split_models"
                splitter.load_split_models(load_dir=load_dir)

            elif choice == '3':
                if not all([splitter.backbone, splitter.neck, splitter.head]):
                    logger.warning("Model belum dimuat! Silakan muat model terlebih dahulu (opsi 2)")
                    continue

                image_path = input("Masukkan path gambar (default: test.png): ").strip() or "test.png"
                if not os.path.exists(image_path):
                    logger.error(f"File gambar tidak ditemukan: {image_path}")
                    continue

                conf = input("Masukkan confidence threshold (0-1, default: 0.25): ").strip() or "0.25"
                try:
                    conf = float(conf)
                    if not 0 <= conf <= 1:
                        raise ValueError("Confidence harus antara 0 dan 1")
                except ValueError as e:
                    logger.error(f"Invalid confidence value: {str(e)}")
                    continue

                logger.info(f"Running prediction on {image_path}...")
                try:
                    splitter.predict(
                        source=image_path,
                        conf=conf,
                        imgsz=640,
                        save=True,
                        project="output"
                    )
                    logger.info(f"Prediction complete. Results saved in 'output' directory")
                except Exception as e:
                    logger.error(f"Prediction failed: {str(e)}")

            elif choice == '4':
                splitter.unload_models()
                logger.info("Model berhasil dibongkar dari memori")

            elif choice == '5':
                logger.info("Program selesai")
                break

            else:
                logger.warning("Pilihan tidak valid! Masukkan angka 1-5")

        except Exception as e:
            logger.error(f"Terjadi kesalahan: {str(e)}")
            logger.debug("Error detail:", exc_info=True)

if __name__ == "__main__":
    run_interactive()
