"""
Alternative Memory Optimization Strategies for TF-IDF Data
"""

def optimize_tfidf_memory():
    """
    Optimize TF-IDF matrices and related data structures for memory efficiency
    """
    
    strategies = {
        "1. Sparse Matrix Optimization": {
            "description": "Your sparse_matrix.npz (204MB) can be further optimized",
            "techniques": [
                "Use float32 instead of float64 (50% memory reduction)",
                "Increase sparsity threshold (remove very low TF-IDF values)",
                "Use block sparse format for better compression"
            ],
            "code_example": """
from scipy.sparse import load_npz, save_npz
import numpy as np

# Load and optimize sparse matrix
matrix = load_npz("Dataset/sparse_matrix.npz")
print(f"Original: {matrix.data.nbytes / 1024**2:.2f} MB")

# Convert to float32 (50% memory reduction)
matrix.data = matrix.data.astype(np.float32)

# Remove very low values (increase sparsity)
threshold = 0.001  # Adjust based on your needs
matrix.data[matrix.data < threshold] = 0
matrix.eliminate_zeros()

save_npz("Dataset/sparse_matrix_optimized.npz", matrix)
print(f"Optimized: {matrix.data.nbytes / 1024**2:.2f} MB")
"""
        },
        
        "2. Dictionary-based TF-IDF": {
            "description": "Replace matrix with dictionary for dynamic loading",
            "techniques": [
                "Only load relevant documents for each query",
                "Use compressed dictionaries",
                "Implement lazy loading"
            ],
            "code_example": """
import pickle
from collections import defaultdict

def create_document_tfidf_dict(tf_idf_matrix, unique_df):
    '''Create dictionary mapping document ID to TF-IDF vector'''
    doc_tfidf = {}
    for i, doc_id in enumerate(unique_df['Id']):
        # Only store non-zero values
        row = tf_idf_matrix[i]
        non_zero_indices = row.nonzero()[1]
        non_zero_values = row.data
        doc_tfidf[doc_id] = dict(zip(non_zero_indices, non_zero_values))
    
    return doc_tfidf

# Usage in Streamlit
@st.cache_data
def load_tfidf_dict():
    with open('Dataset/doc_tfidf_dict.pkl', 'rb') as f:
        return pickle.load(f)
"""
        },
        
        "3. Hierarchical Loading": {
            "description": "Load data in stages based on query complexity",
            "techniques": [
                "Load basic data first",
                "Load TF-IDF data only when needed", 
                "Use query-specific subsets"
            ],
            "code_example": """
class HierarchicalLoader:
    def __init__(self):
        self.basic_data = None
        self.tfidf_data = None
        
    @st.cache_data
    def load_basic_data(self):
        if self.basic_data is None:
            # Load only essential columns
            self.basic_data = pd.read_csv("Dataset/basic_data.csv")
        return self.basic_data
    
    def load_tfidf_data(self):
        if self.tfidf_data is None:
            # Load only when ML features are needed
            self.tfidf_data = load_npz("Dataset/sparse_matrix_optimized.npz")
        return self.tfidf_data
"""
        },
        
        "4. Column Reduction": {
            "description": "Remove unnecessary columns from your dataset",
            "techniques": [
                "Keep only essential columns for your chatbot",
                "Remove intermediate processing columns",
                "Use column-specific loading"
            ],
            "essential_columns": [
                "Id", "Score_question", "Title", "Body_question", 
                "Score_answer", "Body_answer"
            ],
            "removable_columns": [
                "Tag (if not used for filtering)",
                "Unnamed columns",
                "Processing artifacts"
            ]
        }
    }
    
    return strategies

def generate_memory_report():
    """Generate a memory optimization report"""
    report = """
# Memory Optimization Report for StackBot

## Current Memory Usage (Estimated)
- half_cleaned.csv: 371MB
- sparse_matrix.npz: 204MB  
- Other files: ~50MB
- **Total: ~625MB** (within 1GB limit but tight)

## Optimization Priority (Impact vs Effort)

### High Impact, Low Effort:
1. **Data Type Optimization** (10-30% reduction)
   - Convert int64 → int32/int16
   - Use categorical for repeated strings
   
2. **Remove Unused Columns** (10-50% reduction)
   - Drop columns not needed for inference
   - Keep only essential data

### High Impact, Medium Effort:
3. **Text Deduplication** (20-60% reduction)
   - Map duplicate texts to IDs
   - Store unique texts separately
   
4. **Sparse Matrix Optimization** (30-50% reduction)
   - float64 → float32
   - Increase sparsity threshold

### Medium Impact, High Effort:
5. **Dynamic Loading** (Memory cap)
   - Load data on demand
   - Query-specific subsets
   
6. **External Storage** (Unlimited)
   - Use cloud storage for large matrices
   - Load only needed portions

## Recommended Implementation Order:
1. Run memory_optimizer.py on your dataset
2. Test optimized dataset in Streamlit
3. Implement sparse matrix optimization if needed
4. Add dynamic loading for future scaling
"""
    
    with open("memory_optimization_report.md", 'w') as f:
        f.write(report)
    
    print("Generated memory_optimization_report.md")

if __name__ == "__main__":
    strategies = optimize_tfidf_memory()
    generate_memory_report()
    
    print("Memory optimization strategies documented!")
    print("Check memory_optimization_report.md for detailed analysis")