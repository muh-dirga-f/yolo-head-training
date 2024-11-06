from ultralytics import YOLO
import torch
import logging
import os

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetectorTrainer:
    def __init__(self, model_type='yolov8m.pt'):
        """
        Inisialisasi trainer dengan model YOLOv8 yang dipilih
        Hanya backbone yang akan di-freeze, neck dan head akan dilatih
        Args:
            model_type: Tipe model YOLOv8 ('yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
        """
        self.model_type = model_type
        self.backbone_layers = 10  # Jumlah layer backbone YOLOv8

    def prepare_model(self):
        """
        Memuat model YOLOv8 pretrained dan mempersiapkan untuk training
        """
        logger.info(f"Memuat model pretrained {self.model_type}...")

        # Load pretrained model
        self.model = YOLO(self.model_type)

        # Freeze hanya backbone
        model_list = self.model.model.model
        for i in range(self.backbone_layers):
            for param in model_list[i].parameters():
                param.requires_grad = False

        logger.info("Model siap untuk training (neck dan head layer trainable)")

    def train(self, data_yaml, epochs=100, imgsz=640, batch=16):
        """
        Melatih head model dengan dataset yang diberikan
        Args:
            data_yaml: Path ke file konfigurasi dataset
            epochs: Jumlah epochs untuk training
            imgsz: Ukuran gambar input
            batch: Ukuran batch
        """
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"File konfigurasi dataset tidak ditemukan: {data_yaml}")

        logger.info(f"Memulai training dengan {epochs} epochs...")

        # Training dengan ultralytics API
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project='head_training',
            name='exp',
            exist_ok=True
        )

        # Simpan neck dan head yang telah dilatih
        trained_neck_head = torch.nn.Sequential(*self.model.model.model[self.backbone_layers:])
        os.makedirs('trained_models', exist_ok=True)
        torch.save(trained_neck_head, 'trained_models/trained_neck_head.pt')
        logger.info("Training selesai. Neck dan head model disimpan di trained_models/trained_neck_head.pt")

        return results

def main():
    # Inisialisasi trainer dengan model YOLOv8m
    trainer = DetectorTrainer('yolov8m.pt')

    # Persiapkan model
    trainer.prepare_model()

    # Mulai training
    trainer.train(
        data_yaml='/content/custom_dataset.yaml',  # Ganti dengan path dataset Anda
        epochs=10,
        imgsz=640,
        batch=25
    )

if __name__ == "__main__":
    main()
