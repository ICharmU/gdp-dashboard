# Memory Optimization Implementation Guide

## 🚀 Quick Wins (Implement First)

### 1. Column Reduction (Immediate 20-50% savings)
```python
# Load only essential columns
essential_cols = ['Id', 'Title', 'Score_question', 'Score_answer', 'Body_question', 'Body_answer']
df = pd.read_csv("Dataset/half_cleaned.csv", usecols=essential_cols)
```

### 2. Data Type Optimization (10-30% savings) 
```python
# Optimize integer types
df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer')
df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
```

### 3. Use Streamlit Caching
```python
@st.cache_data
def load_data():
    return pd.read_csv("Dataset/half_cleaned.csv", usecols=essential_cols)
```

## 🎯 Advanced Optimizations

### 4. Text Deduplication (20-60% savings)
Your dataset likely has duplicate questions/answers. The memory_optimizer.py script will:
- Find unique questions and answers
- Replace text with numeric IDs
- Store mappings separately

text_cols = ['Title', 'Body_question', 'Body_answer']
for col in text_cols:
    df[col] = df[col].astype('category')

### 5. Sparse Matrix Optimization (30-50% savings)
```python
import numpy as np
from scipy.sparse import load_npz, save_npz

# Load and optimize your TF-IDF matrix
matrix = load_npz("Dataset/sparse_matrix.npz")
matrix.data = matrix.data.astype(np.float32)  # 50% memory reduction
matrix.data[matrix.data < 0.001] = 0  # Increase sparsity
matrix.eliminate_zeros()
save_npz("Dataset/sparse_matrix_optimized.npz", matrix)
```

## 📊 Your Current Situation Analysis

**Files by Memory Impact:**
1. **cleaned.csv (764MB)** - Your biggest problem
2. **half_cleaned.csv (371MB)** - Good start, but still large  
3. **sparse_matrix.npz (204MB)** - Optimizable to ~100MB
4. **Other files (~50MB)** - Not critical

**Recommended Strategy:**
1. Use `streamlit_app_optimized.py` (loads only essential data)
2. Run `memory_optimizer.py` to create deduplicated dataset
3. Optimize sparse matrix if using ML features
4. Monitor memory usage with sidebar metrics

## 🔧 Implementation Steps

1. **Test Current Memory Usage:**
   ```bash
   streamlit run streamlit_app_optimized.py
   ```

2. **Create Optimized Dataset:**
   ```bash
   python memory_optimizer.py
   ```

3. **Switch to Optimized Version:**
   Replace your current data loading with the optimized loader

4. **Monitor Results:**
   Check the sidebar memory metrics in the optimized app

## 💡 Expected Results

| Optimization | Memory Reduction | Effort |
|-------------|------------------|--------|
| Column selection | 10-30% | Low |
| Data type optimization | 10-20% | Low |
| Text deduplication | 30-60% | Medium |
| Sparse matrix optimization | 30-50% | Medium |
| **Combined** | **50-80%** | **Medium** |

With these optimizations, your 625MB dataset could become 125-310MB, well within the 1GB limit with room for processing overhead.

## 🚨 Quick Fix for Today

If you need immediate relief, update your current `streamlit_app.py`:

```python
# Replace this line:
df = pd.read_csv("Dataset/half_cleaned.csv")

# With this:
essential_cols = ['Id', 'Title', 'Score_question', 'Score_answer', 'Body_question', 'Body_answer'] 
df = pd.read_csv("Dataset/half_cleaned.csv", usecols=essential_cols)
df['Id'] = pd.to_numeric(df['Id'], downcast='integer')
df['Score_question'] = pd.to_numeric(df['Score_question'], downcast='integer') 
df['Score_answer'] = pd.to_numeric(df['Score_answer'], downcast='integer')
```

This alone should reduce memory usage by 20-40% immediately!