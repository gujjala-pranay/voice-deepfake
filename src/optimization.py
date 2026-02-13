import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import os
from functools import lru_cache
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelOptimizer:
    """
    Model optimization techniques for faster inference
    """
    def __init__(self):
        self.optimized_models = {}
        
    def quantize_model_dynamic(self, model):
        """
        Dynamic quantization for inference speedup
        """
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU}, 
            dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_model_static(self, model, calibration_loader):
        """
        Static quantization with calibration
        """
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, dict):
                    inputs = (batch['spectral'], batch['mfcc'], batch['phase'])
                else:
                    inputs = batch[0] if len(batch) > 1 else batch
                
                if isinstance(inputs, tuple):
                    model(*inputs)
                else:
                    model(inputs)
        
        quantized_model = torch.quantization.convert(model, inplace=False)
        return quantized_model
    
    def prune_model(self, model, pruning_ratio=0.2):
        """
        Model pruning for efficiency
        """
        import torch.nn.utils.prune as prune
        
        # Global pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio,
        )
        
        # Remove pruning reparameterization to make it permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        return model
    
    def compile_model(self, model):
        """
        Torch compilation for speedup (PyTorch 2.0+)
        """
        try:
            compiled_model = torch.compile(model, mode='reduce-overhead')
            return compiled_model
        except Exception as e:
            print(f"Compilation failed: {e}")
            return model

class FeatureCache:
    """
    Caching system for feature extraction
    """
    def __init__(self, cache_dir='./feature_cache', max_size=1000):
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.memory_cache = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_file_hash(self, file_path):
        """Generate hash for file"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_cache_path(self, file_hash):
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")
    
    def get_features(self, file_path):
        """Get cached features if available"""
        file_hash = self._get_file_hash(file_path)
        
        # Check memory cache first
        if file_hash in self.memory_cache:
            return self.memory_cache[file_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(file_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                
                # Add to memory cache
                if len(self.memory_cache) < self.max_size:
                    self.memory_cache[file_hash] = features
                
                return features
            except Exception:
                pass
        
        return None
    
    def cache_features(self, file_path, features):
        """Cache extracted features"""
        file_hash = self._get_file_hash(file_path)
        
        # Add to memory cache
        if len(self.memory_cache) < self.max_size:
            self.memory_cache[file_hash] = features
        
        # Save to disk cache
        cache_path = self._get_cache_path(file_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception:
            pass
    
    def clear_cache(self):
        """Clear all caches"""
        self.memory_cache.clear()
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, file))

class ParallelProcessor:
    """
    Parallel processing for feature extraction and inference
    """
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or mp.cpu_count()
        
    def parallel_feature_extraction(self, audio_paths, extract_func):
        """
        Extract features in parallel
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(extract_func, path) for path in audio_paths]
            results = [future.result() for future in futures]
        
        return results
    
    def parallel_inference(self, model, feature_batches):
        """
        Run inference in parallel on multiple batches
        """
        def run_inference(batch):
            with torch.no_grad():
                if isinstance(batch, dict):
                    return model(batch['spectral'], batch['mfcc'], batch['phase'])
                else:
                    return model(batch)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(run_inference, batch) for batch in feature_batches]
            results = [future.result() for future in futures]
        
        return results

class BatchProcessor:
    """
    Efficient batch processing with dynamic batching
    """
    def __init__(self, max_batch_size=32, memory_threshold_gb=8):
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold_gb * 1024**3  # Convert to bytes
        
    def get_optimal_batch_size(self, sample_input, model):
        """
        Determine optimal batch size based on memory constraints
        """
        model.eval()
        
        # Start with small batch size and increase until memory limit
        batch_size = 1
        while batch_size <= self.max_batch_size:
            try:
                # Create batch
                if isinstance(sample_input, dict):
                    batch = {}
                    for key, value in sample_input.items():
                        batch[key] = value.repeat(batch_size, *([1] * (len(value.shape) - 1)))
                    test_output = model(batch['spectral'], batch['mfcc'], batch['phase'])
                else:
                    batch = sample_input.repeat(batch_size, *([1] * (len(sample_input.shape) - 1)))
                    test_output = model(batch)
                
                # Check memory usage
                memory_used = psutil.virtual_memory().used
                if memory_used > self.memory_threshold:
                    break
                    
                batch_size *= 2
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    raise e
        
        return max(1, batch_size // 2)  # Return half of the max successful batch size
    
    def process_large_dataset(self, dataset, model, batch_size=None):
        """
        Process large dataset with memory management
        """
        if batch_size is None:
            # Get a sample to determine optimal batch size
            sample = dataset[0]
            if isinstance(sample, dict):
                sample_input = sample['spectral']
            else:
                sample_input = sample[0]
            batch_size = self.get_optimal_batch_size(sample_input, model)
        
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            
            # Process batch
            with torch.no_grad():
                if isinstance(batch[0], dict):
                    # Stack dictionaries
                    batch_dict = {}
                    for key in batch[0].keys():
                        batch_dict[key] = torch.stack([item[key] for item in batch])
                    output = model(batch_dict['spectral'], batch_dict['mfcc'], batch_dict['phase'])
                else:
                    # Stack tensors
                    if isinstance(batch[0], (list, tuple)):
                        inputs = torch.stack([item[0] for item in batch])
                    else:
                        inputs = torch.stack(batch)
                    output = model(inputs)
            
            results.append(output)
            
            # Clear cache periodically
            if i % (batch_size * 10) == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return torch.cat(results, dim=0)

class MemoryOptimizer:
    """
    Memory optimization utilities
    """
    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / 1024**3,
            'available_gb': memory.available / 1024**3,
            'percent': memory.percent
        }
    
    @staticmethod
    def monitor_memory_usage(func):
        """Decorator to monitor memory usage"""
        def wrapper(*args, **kwargs):
            start_memory = MemoryOptimizer.get_memory_usage()
            result = func(*args, **kwargs)
            end_memory = MemoryOptimizer.get_memory_usage()
            
            print(f"Memory usage: {start_memory['used_gb']:.2f}GB -> {end_memory['used_gb']:.2f}GB")
            return result
        return wrapper

class InferenceOptimizer:
    """
    Inference optimization techniques
    """
    def __init__(self):
        self.feature_cache = FeatureCache()
        self.parallel_processor = ParallelProcessor()
        self.batch_processor = BatchProcessor()
        
    def optimize_inference_pipeline(self, model, use_quantization=True, use_compilation=True):
        """
        Optimize the entire inference pipeline
        """
        # Model optimization
        if use_quantization:
            model = ModelOptimizer().quantize_model_dynamic(model)
        
        if use_compilation:
            model = ModelOptimizer().compile_model(model)
        
        return model
    
    def fast_inference(self, model, audio_paths, feature_extractor):
        """
        Fast inference with caching and parallel processing
        """
        # Check cache first
        cached_features = []
        uncached_paths = []
        
        for path in audio_paths:
            cached = self.feature_cache.get_features(path)
            if cached is not None:
                cached_features.append(cached)
            else:
                uncached_paths.append(path)
        
        # Extract features for uncached files in parallel
        if uncached_paths:
            new_features = self.parallel_processor.parallel_feature_extraction(
                uncached_paths, feature_extractor
            )
            
            # Cache new features
            for path, features in zip(uncached_paths, new_features):
                self.feature_cache.cache_features(path, features)
                cached_features.append(features)
        
        # Batch inference
        if cached_features:
            # Convert to batch format
            if isinstance(cached_features[0], dict):
                batch = {}
                for key in cached_features[0].keys():
                    batch[key] = torch.stack([feat[key] for feat in cached_features])
                with torch.no_grad():
                    outputs = model(batch['spectral'], batch['mfcc'], batch['phase'])
            else:
                batch = torch.stack(cached_features)
                with torch.no_grad():
                    outputs = model(batch)
            
            return outputs
        else:
            return None

class Profiler:
    """
    Performance profiling utilities
    """
    def __init__(self):
        self.timings = {}
        
    def profile_function(self, func_name):
        """Decorator to profile function execution time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                if func_name not in self.timings:
                    self.timings[func_name] = []
                self.timings[func_name].append(execution_time)
                
                return result
            return wrapper
        return decorator
    
    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}
        for func_name, times in self.timings.items():
            stats[func_name] = {
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_calls': len(times)
            }
        return stats
    
    def print_performance_report(self):
        """Print performance report"""
        stats = self.get_performance_stats()
        print("\n=== Performance Report ===")
        for func_name, stat in stats.items():
            print(f"{func_name}:")
            print(f"  Average: {stat['avg_time']:.4f}s")
            print(f"  Min: {stat['min_time']:.4f}s")
            print(f"  Max: {stat['max_time']:.4f}s")
            print(f"  Calls: {stat['total_calls']}")
            print()

# Utility functions
def create_optimized_model(base_model, optimization_level='medium'):
    """
    Create optimized model based on optimization level
    """
    optimizer = ModelOptimizer()
    
    if optimization_level == 'light':
        model = optimizer.compile_model(base_model)
    elif optimization_level == 'medium':
        model = optimizer.quantize_model_dynamic(base_model)
        model = optimizer.compile_model(model)
    elif optimization_level == 'heavy':
        model = optimizer.quantize_model_dynamic(base_model)
        model = optimizer.pruning_model(model, pruning_ratio=0.1)
        model = optimizer.compile_model(model)
    else:
        model = base_model
    
    return model

def benchmark_model(model, test_data, num_runs=100):
    """
    Benchmark model performance
    """
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            
            if isinstance(test_data, dict):
                output = model(test_data['spectral'], test_data['mfcc'], test_data['phase'])
            else:
                output = model(test_data)
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'avg_inference_time': np.mean(times),
        'std_inference_time': np.std(times),
        'throughput': 1.0 / np.mean(times)
    }
