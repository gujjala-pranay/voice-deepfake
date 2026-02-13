"""
Initialize and save model weights for the Audio Deepfake Detection system.
Creates default (untrained) models for inference testing.
"""

import torch
import os
import sys
from datetime import datetime
from src.model import get_model
from src.ensemble import create_ensemble


def save_model(model, path, model_name):
    """Safely save model state dict with metadata"""

    model = model.cpu()      # ensure CPU compatibility
    model.eval()             # switch to inference mode

    checkpoint = {
        "model_name": model_name,
        "state_dict": model.state_dict(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pytorch_version": torch.__version__
    }

    torch.save(checkpoint, path)


def init_models():
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    print("🔧 Initializing Audio Deepfake Detection Models...")
    print(f"📁 Model directory: {os.path.abspath(model_dir)}\n")

    models_to_create = {
        "enhanced_model.pth": lambda: get_model("enhanced"),
        "lightweight_model.pth": lambda: get_model("lightweight"),
        "ensemble_standard_model.pth": lambda: create_ensemble("standard"),
        "ensemble_multiscale_model.pth": lambda: create_ensemble("multiscale"),
        "ensemble_adaptive_model.pth": lambda: create_ensemble("adaptive"),
    }

    for filename, creator in models_to_create.items():
        print(f"🔹 Creating {filename} ...")

        model = creator()
        save_path = os.path.join(model_dir, filename)

        if os.path.exists(save_path):
            print("   ⚠️ File exists — overwriting")

        save_model(model, save_path, filename.replace(".pth", ""))

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"   ✅ Saved ({size_mb:.2f} MB)\n")

    print("=" * 60)
    print("✅ MODEL INITIALIZATION COMPLETE")
    print("=" * 60)

    print("\n📊 Models Created:")
    print("   • Enhanced (Spectral + MFCC + Phase)")
    print("   • Lightweight (Spectral only)")
    print("   • Standard Ensemble")
    print("   • MultiScale Ensemble")
    print("   • Adaptive Ensemble")

    print("\n🚀 Ready for inference (Streamlit or API)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        init_models()
        print("🎉 Initialization successful!")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during initialization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
