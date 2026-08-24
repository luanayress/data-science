"""Model registry for tracking and managing models."""

import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from ..utils.logger import get_logger
from ..utils.paths import get_models_path

logger = get_logger(__name__)


class ModelRegistry:
    """Manage model storage and metadata."""
    
    def __init__(self, version: str = "v1"):
        """
        Initialize registry.
        
        Args:
            version: Model version (e.g., 'v1', 'v2').
        """
        self.version = version
        self.models_dir = get_models_path(version)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_pipeline(self, pipeline: Any, metadata: Dict) -> Path:
        """Persist the complete inference pipeline and version metadata."""
        pipeline_path = self.models_dir / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path, compress=0)
        self._save_metadata(dict(metadata), self.models_dir)
        logger.info(f"Pipeline saved to {pipeline_path}")
        return pipeline_path

    def load_pipeline(self) -> Any:
        """Load the complete inference pipeline for this version."""
        pipeline_path = self.models_dir / "pipeline.joblib"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        logger.info(f"Pipeline loaded from {pipeline_path}")
        return pipeline
    
    def save_model(
        self,
        model: Any,
        model_name: str = "model",
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save model and metadata to a subdirectory.
        
        Args:
            model: Model object to save.
            model_name: Name of the model (without extension).
            metadata: Dictionary of metadata.
            
        Returns:
            Path to saved model directory.
        """
        # Create model directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model with joblib
        model_path = model_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path, compress=0)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'saved_at': datetime.now().isoformat(),
            'model_type': type(model).__name__,
            'model_file': f"{model_name}.joblib"
        })
        
        self._save_metadata(metadata, model_dir)
        
        return model_dir
    
    def load_model(self, model_name: str = "model") -> Any:
        """
        Load model from disk.
        
        Args:
            model_name: Name of the model (without extension).
            
        Returns:
            Loaded model object.
            
        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        model_dir = self.models_dir / model_name
        model_path = model_dir / f"{model_name}.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = joblib.load(model_path)
        
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def save_scaler(self, scaler: Any, scaler_name: str = "scaler") -> Path:
        """
        Save preprocessing scaler to a subdirectory.
        
        Args:
            scaler: Scaler object.
            scaler_name: Name of the scaler.
            
        Returns:
            Path to saved scaler directory.
        """
        # Create scaler directory
        scaler_dir = self.models_dir / scaler_name
        scaler_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scaler with joblib
        scaler_path = scaler_dir / f"{scaler_name}.joblib"
        joblib.dump(scaler, scaler_path, compress=0)
        
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save metadata
        metadata = {
            'scaler_name': scaler_name,
            'saved_at': datetime.now().isoformat(),
            'scaler_type': type(scaler).__name__,
            'scaler_file': f"{scaler_name}.joblib"
        }
        
        self._save_metadata(metadata, scaler_dir)
        
        return scaler_dir
    
    def load_scaler(self, scaler_name: str = "scaler") -> Any:
        """
        Load preprocessing scaler.
        
        Args:
            scaler_name: Name of the scaler.
            
        Returns:
            Loaded scaler object.
        """
        scaler_dir = self.models_dir / scaler_name
        scaler_path = scaler_dir / f"{scaler_name}.joblib"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Scaler loaded from {scaler_path}")
        
        return scaler
    
    def _save_metadata(self, metadata: Dict, component_dir: Optional[Path] = None) -> None:
        """Save metadata to JSON file in component directory."""
        if component_dir is None:
            component_dir = self.models_dir
        
        metadata_file = component_dir / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {metadata_file}")
    
    def load_metadata(self, component_name: Optional[str] = None) -> Dict:
        """Load metadata from JSON file."""
        if component_name is None:
            metadata_file = self.models_dir / "metadata.json"
        else:
            metadata_file = self.models_dir / component_name / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return {}
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_models(self) -> list:
        """List all saved model components (subdirectories)."""
        if not self.models_dir.exists():
            return []
        
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]
        model_names = [d.name for d in model_dirs]
        return model_names
