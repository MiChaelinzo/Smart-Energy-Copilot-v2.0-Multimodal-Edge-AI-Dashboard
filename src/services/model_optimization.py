"""
Model Optimization Service

Handles model quantization, memory optimization, and edge deployment preparation
for the ERNIE AI model in the Smart Energy Copilot.
"""

import asyncio
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Service for optimizing AI models for edge deployment"""
    
    def __init__(self, model_cache_dir: str = "data/models"):
        self.settings = get_settings()
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization settings
        self.target_memory_mb = 2048  # 2GB target for RDK X5
        self.quantization_dtype = torch.qint8
        self.optimization_level = "aggressive"  # conservative, balanced, aggressive
        
        logger.info(f"Model optimizer initialized with cache dir: {self.model_cache_dir}")
    
    async def optimize_ernie_for_edge(self, 
                                    model_name: str = "ernie-3.0-base-zh",
                                    target_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """Optimize ERNIE model for edge deployment"""
        try:
            if target_size_mb is None:
                target_size_mb = self.target_memory_mb
            
            logger.info(f"Starting ERNIE model optimization for edge deployment")
            logger.info(f"Target size: {target_size_mb}MB")
            
            optimization_result = {
                "model_name": model_name,
                "target_size_mb": target_size_mb,
                "optimization_level": self.optimization_level,
                "start_time": datetime.now(),
                "steps_completed": [],
                "final_size_mb": None,
                "compression_ratio": None,
                "success": False
            }
            
            # Step 1: Load original model
            logger.info("Step 1: Loading original ERNIE model...")
            original_model, tokenizer, config = await self._load_original_model(model_name)
            original_size_mb = await self._calculate_model_size(original_model)
            optimization_result["original_size_mb"] = original_size_mb
            optimization_result["steps_completed"].append("model_loaded")
            
            logger.info(f"Original model size: {original_size_mb:.2f}MB")
            
            # Step 2: Apply quantization
            logger.info("Step 2: Applying dynamic quantization...")
            quantized_model = await self._apply_quantization(original_model)
            quantized_size_mb = await self._calculate_model_size(quantized_model)
            optimization_result["quantized_size_mb"] = quantized_size_mb
            optimization_result["steps_completed"].append("quantization_applied")
            
            logger.info(f"Quantized model size: {quantized_size_mb:.2f}MB")
            
            # Step 3: Prune model if needed
            if quantized_size_mb > target_size_mb:
                logger.info("Step 3: Applying model pruning...")
                pruned_model = await self._apply_pruning(quantized_model, target_size_mb)
                final_model = pruned_model
                optimization_result["steps_completed"].append("pruning_applied")
            else:
                final_model = quantized_model
                optimization_result["steps_completed"].append("pruning_skipped")
            
            # Step 4: Optimize for inference
            logger.info("Step 4: Optimizing for inference...")
            optimized_model = await self._optimize_for_inference(final_model)
            final_size_mb = await self._calculate_model_size(optimized_model)
            optimization_result["final_size_mb"] = final_size_mb
            optimization_result["steps_completed"].append("inference_optimized")
            
            # Step 5: Save optimized model
            logger.info("Step 5: Saving optimized model...")
            model_path = await self._save_optimized_model(
                optimized_model, tokenizer, config, model_name
            )
            optimization_result["model_path"] = str(model_path)
            optimization_result["steps_completed"].append("model_saved")
            
            # Step 6: Validate optimized model
            logger.info("Step 6: Validating optimized model...")
            validation_result = await self._validate_optimized_model(
                optimized_model, tokenizer, original_model
            )
            optimization_result["validation"] = validation_result
            optimization_result["steps_completed"].append("model_validated")
            
            # Calculate final metrics
            optimization_result["compression_ratio"] = original_size_mb / final_size_mb
            optimization_result["size_reduction_percent"] = (
                (original_size_mb - final_size_mb) / original_size_mb * 100
            )
            optimization_result["end_time"] = datetime.now()
            optimization_result["success"] = final_size_mb <= target_size_mb
            
            logger.info(f"Model optimization completed successfully")
            logger.info(f"Final size: {final_size_mb:.2f}MB (target: {target_size_mb}MB)")
            logger.info(f"Compression ratio: {optimization_result['compression_ratio']:.2f}x")
            logger.info(f"Size reduction: {optimization_result['size_reduction_percent']:.1f}%")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            optimization_result["error"] = str(e)
            optimization_result["success"] = False
            return optimization_result
    
    async def _load_original_model(self, model_name: str) -> Tuple[nn.Module, Any, Any]:
        """Load the original ERNIE model"""
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32  # Start with float32 for quantization
            )
            
            # Set to evaluation mode
            model.eval()
            
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Failed to load original model: {e}")
            raise
    
    async def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            total_size_bytes = param_size + buffer_size
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            return total_size_mb
            
        except Exception as e:
            logger.error(f"Failed to calculate model size: {e}")
            return 0.0
    
    async def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to the model"""
        try:
            # Define layers to quantize
            layers_to_quantize = {
                nn.Linear,
                nn.Conv1d,
                nn.Conv2d,
                nn.LSTM,
                nn.GRU
            }
            
            # Apply dynamic quantization
            quantized_model = quantize_dynamic(
                model,
                qconfig_spec=layers_to_quantize,
                dtype=self.quantization_dtype
            )
            
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model  # Return original model if quantization fails
    
    async def _apply_pruning(self, model: nn.Module, target_size_mb: float) -> nn.Module:
        """Apply model pruning to reduce size"""
        try:
            current_size_mb = await self._calculate_model_size(model)
            
            if current_size_mb <= target_size_mb:
                return model
            
            # Calculate required pruning ratio
            target_ratio = target_size_mb / current_size_mb
            pruning_ratio = 1.0 - target_ratio
            
            logger.info(f"Applying {pruning_ratio:.2%} pruning to reach target size")
            
            # Simple magnitude-based pruning
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Prune weights with smallest magnitudes
                        weight = module.weight.data
                        weight_flat = weight.view(-1)
                        
                        # Calculate threshold for pruning
                        k = int(len(weight_flat) * pruning_ratio)
                        if k > 0:
                            threshold = torch.topk(torch.abs(weight_flat), k, largest=False)[0][-1]
                            mask = torch.abs(weight) > threshold
                            module.weight.data *= mask.float()
            
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    async def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize model for inference performance"""
        try:
            # Convert to evaluation mode
            model.eval()
            
            # Apply torch.jit.script for optimization (if compatible)
            try:
                # Create dummy input for tracing
                dummy_input = torch.randint(0, 1000, (1, 128))  # Typical sequence length
                
                # Trace the model
                traced_model = torch.jit.trace(model, dummy_input)
                
                # Optimize the traced model
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                
                logger.info("Applied TorchScript optimization")
                return optimized_model
                
            except Exception as trace_error:
                logger.warning(f"TorchScript optimization failed: {trace_error}")
                
                # Fallback: Apply basic optimizations
                with torch.no_grad():
                    # Fuse operations where possible
                    if hasattr(torch.jit, 'fuse'):
                        try:
                            model = torch.jit.fuse(model)
                            logger.info("Applied operation fusion")
                        except:
                            pass
                
                return model
            
        except Exception as e:
            logger.error(f"Inference optimization failed: {e}")
            return model
    
    async def _save_optimized_model(self, 
                                  model: nn.Module, 
                                  tokenizer: Any, 
                                  config: Any, 
                                  model_name: str) -> Path:
        """Save the optimized model to disk"""
        try:
            # Create model-specific directory
            model_dir = self.model_cache_dir / f"{model_name}_optimized"
            model_dir.mkdir(exist_ok=True)
            
            # Save model
            model_path = model_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save tokenizer
            tokenizer_path = model_dir / "tokenizer"
            tokenizer.save_pretrained(tokenizer_path)
            
            # Save config
            config_path = model_dir / "config.json"
            config.save_pretrained(model_dir)
            
            # Save optimization metadata
            metadata = {
                "model_name": model_name,
                "optimization_date": datetime.now().isoformat(),
                "quantization_dtype": str(self.quantization_dtype),
                "optimization_level": self.optimization_level,
                "model_size_mb": await self._calculate_model_size(model)
            }
            
            metadata_path = model_dir / "optimization_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Optimized model saved to: {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to save optimized model: {e}")
            raise
    
    async def _validate_optimized_model(self, 
                                      optimized_model: nn.Module, 
                                      tokenizer: Any, 
                                      original_model: nn.Module) -> Dict[str, Any]:
        """Validate the optimized model performance"""
        try:
            validation_result = {
                "inference_test": False,
                "output_similarity": 0.0,
                "performance_metrics": {},
                "errors": []
            }
            
            # Test inference capability
            try:
                test_text = "Energy consumption analysis test"
                inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True)
                
                # Test optimized model inference
                with torch.no_grad():
                    start_time = time.time()
                    optimized_output = optimized_model(**inputs)
                    optimized_inference_time = time.time() - start_time
                
                validation_result["inference_test"] = True
                validation_result["performance_metrics"]["inference_time_ms"] = optimized_inference_time * 1000
                
                # Compare with original model if possible
                try:
                    with torch.no_grad():
                        start_time = time.time()
                        original_output = original_model(**inputs)
                        original_inference_time = time.time() - start_time
                    
                    # Calculate output similarity (cosine similarity)
                    if hasattr(optimized_output, 'last_hidden_state') and hasattr(original_output, 'last_hidden_state'):
                        opt_embeddings = optimized_output.last_hidden_state.mean(dim=1)
                        orig_embeddings = original_output.last_hidden_state.mean(dim=1)
                        
                        similarity = torch.cosine_similarity(opt_embeddings, orig_embeddings, dim=1)
                        validation_result["output_similarity"] = float(similarity.mean())
                    
                    validation_result["performance_metrics"]["original_inference_time_ms"] = original_inference_time * 1000
                    validation_result["performance_metrics"]["speedup_ratio"] = original_inference_time / optimized_inference_time
                    
                except Exception as comparison_error:
                    validation_result["errors"].append(f"Comparison failed: {comparison_error}")
                
            except Exception as inference_error:
                validation_result["errors"].append(f"Inference test failed: {inference_error}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                "inference_test": False,
                "output_similarity": 0.0,
                "performance_metrics": {},
                "errors": [str(e)]
            }
    
    async def load_optimized_model(self, model_name: str) -> Tuple[Optional[nn.Module], Optional[Any], Optional[Any]]:
        """Load a previously optimized model"""
        try:
            model_dir = self.model_cache_dir / f"{model_name}_optimized"
            
            if not model_dir.exists():
                logger.warning(f"Optimized model not found: {model_dir}")
                return None, None, None
            
            # Load config
            config = AutoConfig.from_pretrained(model_dir)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            # Load model
            model = AutoModel.from_pretrained(model_dir, config=config)
            
            # Load state dict if available
            model_path = model_dir / "model.pt"
            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
            
            model.eval()
            
            logger.info(f"Loaded optimized model from: {model_dir}")
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Failed to load optimized model: {e}")
            return None, None, None
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of model optimizations"""
        try:
            status = {
                "cache_dir": str(self.model_cache_dir),
                "optimized_models": [],
                "total_cache_size_mb": 0.0
            }
            
            # Scan for optimized models
            for model_dir in self.model_cache_dir.glob("*_optimized"):
                if model_dir.is_dir():
                    metadata_path = model_dir / "optimization_metadata.json"
                    
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Calculate directory size
                        dir_size_mb = sum(
                            f.stat().st_size for f in model_dir.rglob('*') if f.is_file()
                        ) / (1024 * 1024)
                        
                        status["optimized_models"].append({
                            "name": metadata.get("model_name", "unknown"),
                            "optimization_date": metadata.get("optimization_date"),
                            "size_mb": dir_size_mb,
                            "path": str(model_dir)
                        })
                        
                        status["total_cache_size_mb"] += dir_size_mb
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get optimization status: {e}")
            return {
                "cache_dir": str(self.model_cache_dir),
                "optimized_models": [],
                "total_cache_size_mb": 0.0,
                "error": str(e)
            }