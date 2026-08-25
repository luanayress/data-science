"""Test the reorganized model loading"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.models.registry import ModelRegistry

print("🧪 TESTING MODEL REGISTRY")
print("=" * 60)

try:
    registry = ModelRegistry(version="v1")
    print("✓ Registry initialized")
    
    # Test loading model
    print("\n1️⃣  Loading model...")
    model = registry.load_model("model")
    print(f"   ✓ Model loaded: {type(model).__name__}")
    
    # Test loading scaler
    print("\n2️⃣  Loading scaler...")
    scaler = registry.load_scaler("scaler")
    print(f"   ✓ Scaler loaded: {type(scaler).__name__}")
    
    # Test loading metadata
    print("\n3️⃣  Loading metadata...")
    model_meta = registry.load_metadata("model")
    scaler_meta = registry.load_metadata("scaler")
    print(f"   ✓ Model metadata: {model_meta['type']}")
    print(f"   ✓ Scaler metadata: {scaler_meta['type']}")
    
    # List components
    print("\n4️⃣  Listing components...")
    components = registry.list_models()
    print(f"   ✓ Found {len(components)} components: {components}")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
