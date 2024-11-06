from ultralytics import YOLO
import torch
import torch.nn as nn
import logging
import os
from typing import List

class BackboneModule(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        features = []
        for i, module in enumerate(self.modules_list):
            x = module(x)
            if i in [4, 6, 8]:  # Save output from specific layers
                features.append(x.clone())  # Clone to prevent modifications
        return x, features  # No need to reverse, we'll handle order in neck

class NeckModule(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_list = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor, features: List[torch.Tensor]) -> torch.Tensor:
        feature_idx = 0
        for module in self.modules_list:
            if isinstance(module, nn.modules.container.Sequential):
                x = module(x)
            elif hasattr(module, 'forward'):
                if isinstance(module, nn.Conv2d):
                    # Apply convolution directly
                    x = module(x)
                elif hasattr(module, 'd'):  # Concat module
                    if feature_idx < len(features):
                        # Get feature and ensure dimensions match
                        feat = features[feature_idx]
                        if feat.shape[2:] != x.shape[2:]:
                            feat = nn.functional.interpolate(feat, size=x.shape[2:], mode='nearest')
                        # Adjust channel dimension if needed
                        if feat.shape[1] != x.shape[1]:
                            adjust_conv = nn.Conv2d(feat.shape[1], x.shape[1], 1).to(feat.device)
                            feat = adjust_conv(feat)
                        x = torch.cat([x, feat], dim=1)
                        feature_idx += 1
                else:
                    x = module(x)
        return x

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YOLOSplitter:
    def __init__(self):
        """
        Inisialisasi YOLOSplitter tanpa memuat model
        """
        logger.info("Initializing YOLOSplitter without loading model")

        self.model = None
        self.backbone = None
        self.neck = None
        self.head = None

    def split_model(self, model_path=None):
        """
        Memecah model menjadi backbone, neck dan head
        Args:
            model_path: Path ke model YOLOv8 (.pt file) atau None untuk menggunakan pretrained
        """
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading and splitting custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            if model_path:
                logger.warning(f"Model file {model_path} not found, using pretrained yolov8m")
            else:
                logger.info("No model specified, using pretrained yolov8m")
            self.model = YOLO('yolov8m.pt')

        # Mengakses model PyTorch internal
        model = self.model.model

        # Backbone layers (dari input sampai sebelum SPPF)
        backbone_modules = list(model.model[:9])
        self.backbone = BackboneModule(backbone_modules)

        # Neck layers (SPPF sampai sebelum Detect)
        neck_modules = list(model.model[9:-1])
        self.neck = NeckModule(neck_modules)

        # Head (Detect layer)
        self.head = model.model[-1]

        # Simpan names dari model asli ke head
        if hasattr(model, 'names') and model.names:
            self.head.names = model.names

        # Unload YOLO model to free memory
        self.model = None
        torch.cuda.empty_cache()

        return self.backbone, self.neck, self.head

    def save_split_models(self, save_dir='split_models/'):
        """
        Menyimpan model yang telah dipecah
        Args:
            save_dir: Direktori untuk menyimpan model
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Simpan model lengkap, bukan hanya state dict
        torch.save(self.backbone, f'{save_dir}/backbone.pt')
        torch.save(self.neck, f'{save_dir}/neck.pt')
        torch.save(self.head, f'{save_dir}/head.pt')

    def load_split_models(self, load_dir='split_models/'):
        """
        Memuat model yang telah dipecah
        Args:
            load_dir: Direktori tempat model tersimpan
        """
        # Cek keberadaan direktori dan file
        if not os.path.exists(load_dir):
            logger.info(f"Direktori {load_dir} tidak ditemukan.")
            os.makedirs(load_dir, exist_ok=True)
            return

        # Cek kelengkapan file model
        missing_files = [part for part in ['backbone', 'neck', 'head']
                        if not os.path.exists(f'{load_dir}/{part}.pt')]

        if missing_files:
            logger.info(f"File model tidak lengkap ({', '.join(missing_files)} tidak ditemukan). Memecah model...")
            self.split_model()
            self.save_split_models(load_dir)
            logger.info("Model berhasil dipecah dan disimpan")
            return

        # Load model dari file yang sudah ada
        logger.info("Memuat model dari file yang tersimpan...")
        self.backbone = torch.load(f'{load_dir}/backbone.pt')
        self.neck = torch.load(f'{load_dir}/neck.pt')
        self.head = torch.load(f'{load_dir}/head.pt')
        logger.info("Model berhasil dimuat dari file yang tersimpan")

    def merge_model(self):
        """
        Menggabungkan model yang telah dipecah menjadi satu model utuh
        Returns:
            YOLO: Model yang telah digabung
        """
        # Memastikan semua komponen telah dimuat
        if not all([self.backbone, self.neck, self.head]):
            raise RuntimeError("All model components must be loaded before merging")

        # Create a new YOLO model without loading weights
        self.model = YOLO('yolov8n.yaml')  # Just loads architecture
        model_list = self.model.model.model

        # Update backbone layers
        for i in range(9):
            model_list[i] = self.backbone.modules_list[i]

        # Update neck layers
        for i, module in enumerate(self.neck.modules_list):
            model_list[i + 9] = module

        # Update head layer
        model_list[-1] = self.head

        # Preserve original class names if they exist in the head
        if hasattr(self.head, 'names') and self.head.names:
            self.model.model.names = self.head.names

        return self.model

    def predict(self, source, **kwargs):
        """
        Melakukan prediksi menggunakan model yang telah digabung
        Args:
            source: Path ke gambar atau direktori gambar
            **kwargs: Parameter tambahan untuk YOLO predict
        Returns:
            List: Hasil prediksi
        """
        merged_model = self.merge_model()
        results = merged_model.predict(source=source, **kwargs)

        return results

    def unload_models(self):
        """
        Menghapus model dari memori
        """
        self.backbone = None
        self.neck = None
        self.head = None
        torch.cuda.empty_cache()  # Membersihkan GPU memory jika menggunakan CUDA
