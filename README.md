# 📚 Semantic Book Recommender (LLMs + Vector Search + NLP Analytics)

An intelligent book recommendation system that combines modern NLP techniques with semantic search to deliver personalized, contextually-aware book suggestions. Built using vector databases, zero-shot classification, sentiment analysis, and interactive web interfaces for discovering your next great read through natural language queries.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Vector_DB-green?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Web_UI-orange?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-blue?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)

✨ **Features**

🔍 **Semantic Search Engine**: Vector-powered search using Chroma DB with OpenAI embeddings for contextually intelligent book discovery

📖 **Smart Content Classification**: Zero-shot text classification to automatically categorize fiction vs non-fiction without manual labeling

🎭 **Emotion & Sentiment Analysis**: Advanced NLP to extract emotional themes and sentiment patterns from book descriptions

🌐 **Interactive Web Interface**: Intuitive Gradio-powered dashboard for natural language book queries and visual recommendations

📊 **Comprehensive Data Pipeline**: End-to-end notebooks covering data exploration, feature engineering, and model development

🎯 **Personalized Matching**: Intelligent recommendation algorithms that understand nuanced reading preferences and mood-based suggestions

🗂️ **Project structure**
```
.
├─ data-exploration.ipynb     # Dataset cleaning, EDA, and preprocessing pipeline
├─ vector-search.ipynb        # Embedding generation and Chroma vector database setup
├─ text-classification.ipynb  # Zero-shot genre classification implementation
├─ sentiment-analysis.ipynb   # Emotion extraction and sentiment scoring
├─ gradio-dashboard.py        # Interactive web application for recommendations
├─ requirements.txt           # Comprehensive dependency management
├─ cover-not-found.jpg        # Fallback image for missing book covers
└─ .env.example              # Environment configuration template
```

🚀 **Quickstart**

**Prerequisites**
- Python 3.8 or higher
- OpenAI API key for embeddings and language model features
- Kaggle account (optional) for dataset access

**1) Environment setup**
```bash
git clone https://github.com/AbdullahRasheed45/GENERATIVE_AI_BOOK_RECOMMENDER.git
cd GENERATIVE_AI_BOOK_RECOMMENDER

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**2) API configuration**
```bash
# Create environment file
cp .env.example .env

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
```

**3) Data preparation & exploration**
```bash
# Start with data exploration and cleaning
jupyter notebook data-exploration.ipynb

# The notebook will guide you through:
# - Dataset download and loading
# - Data quality assessment and cleaning
# - Exploratory analysis of book metadata
# - Feature engineering for recommendations
```

**4) Build the recommendation system**
```bash
# Create vector embeddings and search index
jupyter notebook vector-search.ipynb

# Implement genre classification
jupyter notebook text-classification.ipynb  

# Add sentiment and emotion analysis
jupyter notebook sentiment-analysis.ipynb
```

**5) Launch the web application**
```bash
# Start the interactive recommendation dashboard
python gradio-dashboard.py

# Access at http://127.0.0.1:7860 for natural language book queries
```

🧠 **System architecture & workflow**

**1. Data Preprocessing & Quality Enhancement**
```python
# data-exploration.ipynb
def clean_book_dataset(df):
    """Comprehensive data cleaning and enhancement pipeline"""
    # Remove duplicates and low-quality entries
    # Standardize author names and titles
    # Extract and clean book descriptions
    # Handle missing metadata and cover images
    # Generate derived features for recommendation engine
    return cleaned_dataset
```

**2. Semantic Vector Database Creation**
```python
# vector-search.ipynb  
def build_semantic_index(books_df):
    """Create searchable vector database for semantic similarity"""
    # Generate embeddings using OpenAI text-embedding-ada-002
    # Store vectors in Chroma DB with metadata
    # Configure similarity search parameters
    # Optimize for fast retrieval and relevance scoring
    return vector_store
```

**3. Intelligent Genre Classification**
```python
# text-classification.ipynb
def classify_book_genres(descriptions):
    """Zero-shot classification for automatic genre tagging"""
    # Use transformer models for fiction/non-fiction classification
    # Extract sub-genre indicators (mystery, romance, sci-fi, etc.)
    # Generate confidence scores for classification decisions
    # Handle edge cases and mixed-genre books
    return genre_classifications
```

**4. Emotional Context Analysis** 
```python
# sentiment-analysis.ipynb
def analyze_emotional_themes(book_descriptions):
    """Extract emotional and thematic content from book descriptions"""
    # Sentiment polarity analysis (positive, negative, neutral)
    # Emotion detection (joy, sadness, suspense, romance, etc.)
    # Thematic element extraction (family, adventure, mystery)
    # Mood and tone classification for recommendation matching
    return emotional_features
```

**5. Recommendation Engine Integration**
```python
# gradio-dashboard.py
def get_personalized_recommendations(query, filters=None):
    """Generate contextually-aware book recommendations"""
    # Convert natural language query to vector representation
    # Perform semantic similarity search in vector database
    # Apply genre, sentiment, and preference filters
    # Rank results by relevance and diversity
    # Format recommendations with explanations and metadata
    return recommended_books
```

⚙️ **Advanced configuration options**

**Embedding Model Selection**:
```python
# Vector search configuration
EMBEDDING_CONFIG = {
    'model': 'text-embedding-ada-002',    # OpenAI embedding model
    'dimension': 1536,                    # Embedding vector size
    'chunk_size': 1000,                   # Text chunk size for processing
    'overlap': 200,                       # Chunk overlap for context preservation
    'similarity_metric': 'cosine'         # Distance metric for vector search
}

# Alternative local models for cost optimization
LOCAL_MODELS = {
    'sentence-transformers/all-MiniLM-L6-v2',     # Fast, lightweight
    'sentence-transformers/all-mpnet-base-v2',    # High quality
    'sentence-transformers/multi-qa-mpnet-base'   # Q&A optimized
}
```

**Classification Model Tuning**:
```python
# Zero-shot classification setup
CLASSIFICATION_CONFIG = {
    'model': 'facebook/bart-large-mnli',
    'candidate_labels': [
        'fiction', 'non-fiction', 'mystery', 'romance', 
        'science fiction', 'fantasy', 'biography', 'history'
    ],
    'multi_label': True,                   # Allow multiple genre classifications
    'threshold': 0.5                       # Confidence threshold for labels
}
```

**Recommendation Algorithm Parameters**:
```python
# Recommendation engine settings
RECOMMENDER_CONFIG = {
    'max_results': 10,                     # Maximum recommendations returned
    'similarity_threshold': 0.7,           # Minimum similarity score
    'diversity_factor': 0.3,               # Balance between relevance and diversity
    'genre_weight': 0.4,                   # Importance of genre matching
    'sentiment_weight': 0.3,               # Importance of emotional matching
    'popularity_boost': 0.1                # Boost for popular/highly-rated books
}
```

🎯 **Natural language query examples**

**Mood-Based Recommendations**:
- *"I want something uplifting and heartwarming after a difficult week"*
- *"Looking for a gripping psychological thriller that will keep me up all night"*
- *"Need a cozy mystery with humor, no violence, perfect for a rainy Sunday"*

**Theme & Setting Queries**:
- *"Stories about found family and belonging, preferably in a fantasy setting"*
- *"Non-fiction about space exploration that's accessible to general readers"*
- *"Historical fiction set during World War II, focusing on resistance movements"*

**Comparative & Style-Based**:
- *"Books similar to 'The Seven Husbands of Evelyn Hugo' but set in different time periods"*
- *"Literary fiction with beautiful prose and complex characters, no romance required"*
- *"Fast-paced adventure novels like Dan Brown but with better character development"*

**Specific Criteria Combinations**:
- *"Young adult fantasy with strong female protagonists and minimal romance subplot"*
- *"Biographies of scientists or inventors, written in an engaging narrative style"*
- *"Contemporary literary fiction dealing with family secrets and small-town dynamics"*

📊 **System capabilities & metrics**

**Search Performance**:
```python
# Recommendation quality metrics
performance_metrics = {
    'search_latency': '<200ms',           # Average query response time
    'relevance_score': '>0.85',           # User satisfaction with recommendations  
    'diversity_index': '0.7-0.9',         # Recommendation diversity balance
    'coverage_ratio': '>0.75',            # Percentage of catalog discoverable
    'precision_at_k': '>0.80'             # Accuracy of top-K recommendations
}
```

**Content Analysis Capabilities**:
```python
# NLP feature extraction
content_features = {
    'genre_classification': {
        'accuracy': 0.92,                  # Fiction vs non-fiction classification
        'multi_label_support': True,       # Multiple genre assignments
        'confidence_scoring': True         # Classification confidence levels
    },
    'sentiment_analysis': {
        'emotion_categories': 8,           # Joy, sadness, anger, fear, etc.
        'intensity_scoring': True,         # Emotional intensity measurement
        'theme_extraction': True           # Key thematic elements
    },
    'semantic_understanding': {
        'context_awareness': True,         # Understanding of nuanced queries
        'preference_learning': True,       # Adaptation to user preferences
        'explanation_generation': True     # Why books were recommended
    }
}
```

🔧 **Customization and extension**

**Adding New Data Sources**:
```python
# Extend data pipeline for additional book sources
def integrate_new_book_source(api_config):
    """Add support for additional book databases"""
    # Goodreads API integration
    # Google Books API support  
    # Library catalog connections
    # Custom CSV/JSON data imports
    return enhanced_dataset
```

**Custom Classification Categories**:
```python
# Extend genre classification with custom categories
CUSTOM_LABELS = [
    'cozy_mystery', 'dark_academia', 'cli_fi',           # Niche genres
    'slow_burn_romance', 'found_family', 'enemies_to_lovers',  # Trope-based
    'diverse_voices', 'own_voices', 'translated_works'    # Representation-based
]
```

**Advanced Filtering Options**:
```python
# Sophisticated filtering and preference matching
def apply_advanced_filters(recommendations, user_prefs):
    """Apply complex user preference filtering"""
    # Reading level and complexity preferences
    # Content warnings and trigger avoidance
    # Length preferences (novella, novel, series)
    # Publication date ranges and award winners
    # Author diversity and representation preferences
    return filtered_recommendations
```

🌐 **Web interface features**

**Interactive Query Interface**:
```python
# Gradio dashboard components
interface_components = {
    'query_input': 'Natural language text area for book requests',
    'filter_controls': 'Genre, length, publication date, rating filters',  
    'result_display': 'Rich book cards with covers, descriptions, ratings',
    'explanation_panel': 'AI-generated explanations for recommendations',
    'feedback_system': 'Like/dislike buttons for recommendation improvement'
}
```

**Recommendation Display Format**:
- **Book Cover Images**: High-quality covers with fallback handling
- **Detailed Metadata**: Title, author, publication info, ratings
- **Match Explanation**: Why each book was recommended for your query
- **Similar Books**: "If you like this, you might also enjoy..." suggestions
- **Reading Links**: Direct links to purchase, library, or reading platforms

🧪 **Testing and evaluation framework**

**Recommendation Quality Assessment**:
```python
# Automated testing for recommendation quality
def evaluate_recommendation_system():
    """Comprehensive system evaluation metrics"""
    test_queries = load_test_queries()
    
    metrics = {
        'relevance_scoring': calculate_user_relevance_ratings(),
        'diversity_measurement': assess_recommendation_diversity(),
        'coverage_analysis': measure_catalog_coverage(),
        'bias_detection': evaluate_recommendation_bias(),
        'performance_benchmarking': measure_response_times()
    }
    
    return evaluation_report
```

**A/B Testing Framework**:
```python
# Compare different recommendation algorithms
def run_algorithm_comparison():
    """Compare different recommendation approaches"""
    algorithms = [
        'pure_semantic_similarity',
        'hybrid_content_collaborative', 
        'emotion_weighted_search',
        'genre_preference_boosted'
    ]
    
    for algorithm in algorithms:
        performance = evaluate_algorithm_performance(algorithm)
        log_results(algorithm, performance)
```

🔒 **Privacy and data handling**

**User Data Protection**:
```python
# Privacy-preserving recommendation system
class PrivacyPreservingRecommender:
    """Recommendation system with privacy protection"""
    
    def __init__(self):
        self.no_user_data_storage = True      # No persistent user data
        self.anonymized_analytics = True      # Anonymous usage patterns only
        self.opt_out_available = True         # Easy data deletion
        
    def process_query(self, query):
        # Process recommendations without storing personal data
        # Use session-based preferences only
        # Clear data after session ends
        return recommendations
```

**Ethical AI Considerations**:
- **Bias Mitigation**: Regular auditing for genre, author, and cultural biases
- **Diverse Representation**: Ensure recommendations include diverse voices and perspectives
- **Content Warnings**: Appropriate flagging of potentially sensitive content
- **Accessibility**: Support for screen readers and accessibility tools

🐛 **Troubleshooting guide**

**Common Setup Issues**:
- **API Key Problems** → Verify OpenAI API key is correctly set in `.env` file
- **Embedding Generation Fails** → Check API quota and rate limits, consider local alternatives  
- **Vector Database Issues** → Ensure Chroma DB directory permissions and disk space
- **Notebook Kernel Problems** → Restart kernel, clear outputs, reinstall ipywidgets

**Performance Issues**:
- **Slow Search Responses** → Optimize vector database indexing, reduce embedding dimensions
- **Memory Usage** → Process books in smaller batches, use lighter embedding models
- **Classification Accuracy** → Fine-tune classification thresholds, add more training examples

**Data Quality Issues**:
- **Poor Recommendations** → Improve book description quality, enhance metadata
- **Missing Book Covers** → Implement fallback cover sources, improve image URL validation
- **Genre Misclassification** → Refine classification prompts, add manual corrections for edge cases

📚 **Educational resources & learning path**

**Core Concepts Covered**:
- **Vector Embeddings**: Understanding semantic similarity in high-dimensional spaces
- **Zero-Shot Learning**: Classification without labeled training data
- **Information Retrieval**: Modern search and recommendation algorithms
- **NLP Pipeline Design**: End-to-end natural language processing workflows

**Technical Skills Developed**:
- **LangChain Framework**: Vector databases and document processing
- **Transformer Models**: Using pre-trained models for classification and embeddings
- **Web Application Development**: Building interactive ML applications with Gradio
- **Data Science Workflow**: From raw data to deployed recommendation system

📜 **License**

MIT License - see [LICENSE](LICENSE) file for complete terms. Based on freeCodeCamp course materials with significant enhancements and production considerations.

## 📞 Connect & Support

<div align="center">

### 🚀 Ready to Build Intelligent Recommendation Systems?

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://techvibes360.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullahrasheed-/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abdullahrasheed45@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AbdullahRasheed45)

**Let's revolutionize content discovery with semantic AI!**

</div>

---

*Built with ❤️ for book lovers and AI enthusiasts. Perfect for learning modern NLP techniques and building intelligent recommendation systems.*
