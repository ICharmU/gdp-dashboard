"""
NLTK Setup Script - ensures required NLTK data is available for Streamlit deployment
"""

def setup_nltk():
    """Download required NLTK data for Streamlit Cloud deployment"""
    try:
        import nltk
        import ssl
        
        # Handle SSL issues that sometimes occur in cloud environments
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Download required NLTK data
        print("📥 Downloading NLTK data...")
        
        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
            print("✅ Stopwords already available")
        except LookupError:
            print("📥 Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            print("✅ Stopwords downloaded")
        
        # Download punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ Punkt tokenizer already available")
        except LookupError:
            print("📥 Downloading punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            print("✅ Punkt tokenizer downloaded")
        
        # Download wordnet lemmatizer
        try:
            nltk.data.find('corpora/wordnet')
            print("✅ WordNet already available")
        except LookupError:
            print("📥 Downloading wordnet...")
            nltk.download('wordnet', quiet=True)
            print("✅ WordNet downloaded")
            
        print("🎉 NLTK setup complete!")
        return True
        
    except ImportError:
        print("❌ NLTK not available")
        return False
    except Exception as e:
        print(f"❌ Error setting up NLTK: {e}")
        return False

if __name__ == "__main__":
    setup_nltk()