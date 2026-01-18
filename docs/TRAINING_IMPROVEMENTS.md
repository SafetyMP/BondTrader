# Model Training Improvements

## Current Bottlenecks Identified

### 1. **Sequential Training** ⏱️ **Highest Impact**
- **Issue**: All 11 models/steps train sequentially (one after another)
- **Impact**: ~15-30 minutes total training time
- **Solution**: Parallelize independent models (steps 1-8 can run in parallel groups)

**Potential Speedup**: 40-60% reduction in total time (from ~25 min to ~10-15 min)

### 2. **Inefficient Data Conversion** 🔄
- **Issue**: `_convert_to_bonds()` called 3 times (train/validation/test), reconstructing all bonds
- **Impact**: Wasted computation time
- **Solution**: Cache converted bonds or optimize conversion

**Potential Speedup**: 5-10% reduction

### 3. **No Progress Tracking** 📊
- **Issue**: No ETA, progress bars, or time estimates
- **Impact**: Difficult to monitor training progress
- **Solution**: Add `tqdm` progress bars and time tracking

### 4. **Hardcoded Sample Sizes** 🎯
- **Issue**: Evaluation steps use fixed samples (`self.test_bonds[:100]`, `self.train_bonds[:100]`)
- **Impact**: Inconsistent evaluation across different dataset sizes
- **Solution**: Use percentage-based or configurable sampling

### 5. **No Early Stopping** ⏹️
- **Issue**: Models don't use early stopping for convergence
- **Impact**: Longer training times without benefit
- **Solution**: Add early stopping callbacks for iterative models

### 6. **No Resume Capability** 🔄
- **Issue**: Training must restart from scratch if interrupted
- **Impact**: Lost progress on failure
- **Solution**: Checkpoint intermediate models

### 7. **Memory Inefficiency** 💾
- **Issue**: All bonds loaded into memory simultaneously
- **Impact**: High memory usage
- **Solution**: Lazy loading or batching for large datasets

### 8. **Redundant Computations** 🔁
- **Issue**: Fair values recalculated multiple times for same bonds
- **Impact**: Wasted CPU cycles
- **Solution**: Cache fair value calculations

## Recommended Improvements (Priority Order)

### High Priority (Implement First)

#### 1. **Parallel Training** 🚀
**Impact**: 40-60% speedup

Train independent models in parallel:
- **Group 1** (can run in parallel): Models 1-4 (Basic ML, Enhanced ML, Advanced ML, AutoML)
- **Group 2** (can run in parallel): Models 5-8 (Regime, Factor, Tail Risk, Bayesian)
- **Sequential**: Steps 9-11 (Evaluation, Drift Detection, Tuning - depends on earlier results)

**Implementation**:
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

def train_models_parallel(trainer, models_to_train):
    """Train independent models in parallel"""
    with ProcessPoolExecutor(max_workers=min(len(models_to_train), mp.cpu_count())) as executor:
        futures = {executor.submit(train_func, args): name 
                  for name, train_func, args in models_to_train}
        # Wait for completion
```

#### 2. **Progress Tracking with ETA** 📊
**Impact**: Better UX, easier monitoring

Add `tqdm` progress bars:
```python
from tqdm import tqdm
import time

class ModelTrainerWithProgress:
    def train_with_progress(self):
        steps = [
            ("Basic ML Adjuster", self._train_basic_ml),
            ("Enhanced ML Adjuster", self._train_enhanced_ml),
            # ... etc
        ]
        
        with tqdm(total=len(steps), desc="Training Progress") as pbar:
            for step_name, train_func in steps:
                start_time = time.time()
                result = train_func()
                elapsed = time.time() - start_time
                pbar.set_description(f"{step_name} ({elapsed:.1f}s)")
                pbar.update(1)
```

#### 3. **Caching Converted Bonds** 💾
**Impact**: 5-10% speedup

```python
from functools import lru_cache
import joblib

class ModelTrainer:
    @property
    def train_bonds(self):
        if not hasattr(self, '_train_bonds_cache'):
            self._train_bonds_cache = self._convert_to_bonds(self.dataset['train'])
        return self._train_bonds_cache
```

### Medium Priority

#### 4. **Early Stopping**
Add early stopping to iterative models (Gradient Boosting, Neural Networks):
```python
from sklearn.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

#### 5. **Checkpointing**
Save intermediate models:
```python
checkpoint_dir = 'training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# After each model
joblib.dump(model, f'{checkpoint_dir}/{model_name}.joblib')
```

#### 6. **Fair Value Caching**
Cache expensive calculations:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def cached_fair_value(bond_id, coupon_rate, maturity_date_str, ...):
    # Calculate fair value
    return fair_value
```

### Low Priority (Nice to Have)

#### 7. **Resume from Checkpoint**
Load existing checkpoints and continue:
```python
def resume_training(checkpoint_dir):
    """Resume training from last checkpoint"""
    checkpoints = glob.glob(f'{checkpoint_dir}/*.joblib')
    # Determine last completed model
    # Resume from there
```

#### 8. **Memory-Optimized Batching**
Process bonds in batches:
```python
def train_in_batches(bonds, batch_size=1000):
    for i in range(0, len(bonds), batch_size):
        batch = bonds[i:i+batch_size]
        # Train on batch
```

## Implementation Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add progress tracking with `tqdm`
2. ✅ Add timing per step
3. ✅ Cache converted bonds

### Phase 2: Performance (2-4 hours)
1. ⬜ Implement parallel training for independent models
2. ⬜ Add fair value caching
3. ⬜ Optimize bond conversion

### Phase 3: Reliability (2-3 hours)
1. ⬜ Add checkpointing
2. ⬜ Add resume capability
3. ⬜ Better error handling

### Phase 4: Advanced (Optional)
1. ⬜ Early stopping for iterative models
2. ⬜ Memory-optimized batching
3. ⬜ Distributed training (for very large datasets)

## Expected Performance Gains

| Improvement | Time Savings | Difficulty |
|------------|--------------|------------|
| Parallel Training | 40-60% (10-15 min) | Medium |
| Progress Tracking | N/A (UX only) | Easy |
| Bond Conversion Caching | 5-10% (1-2 min) | Easy |
| Fair Value Caching | 5-10% (1-2 min) | Medium |
| Early Stopping | 10-20% (2-5 min) | Medium |
| Checkpointing | N/A (Reliability) | Easy |

**Total Potential Speedup**: 50-70% reduction in training time
- **Current**: ~20-30 minutes
- **Improved**: ~8-12 minutes

## Compatibility Notes

- Parallel training requires models to be thread-safe (sklearn models are generally safe)
- Some models may share state (valuator), need careful handling
- Checkpointing adds disk I/O overhead but provides safety
- Progress tracking adds minimal overhead (~1-2%)

## Next Steps

1. Review and approve this improvement plan
2. Start with Phase 1 (Quick Wins) - immediate visible improvements
3. Measure actual performance gains after each phase
4. Iterate based on results
