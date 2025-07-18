import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import os
from datetime import datetime
import re
from scraper_integrated import IMDBScraper

# Disable TensorFlow for now due to compatibility issues
TENSORFLOW_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f4e79;
        margin-bottom: 2rem;
    }
    .verdict-positive {
        background: linear-gradient(90deg, #00C851, #007E33);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .verdict-negative {
        background: linear-gradient(90deg, #ff4444, #CC0000);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .verdict-mixed {
        background: linear-gradient(90deg, #ffbb33, #FF8800);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize model and tokenizer
@st.cache_resource
def load_sentiment_model():
    """Load the pre-trained sentiment analysis model"""
    if not TENSORFLOW_AVAILABLE:
        return None, None
        
    try:
        # Import TensorFlow only when needed
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import Tokenizer
        import pickle
        
        # Check if the model file exists
        model_path =  'model/best_model.h5'
        if not os.path.exists(model_path):
            st.info("Model file 'best_model.h5' not found. Using advanced keyword-based analysis.")
            return None, None
        
        # Load the model
        else:
            model = load_model(model_path)
        
        # Try to load tokenizer if it exists
        tokenizer = None
        if os.path.exists('tokenizer.pickle'):
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            # Create a basic tokenizer if none exists
            st.warning("Tokenizer not found. Using basic tokenizer. Results may vary.")
            tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
            # Note: This will need to be fitted on your training data
        
        st.success("Custom model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        st.warning(f"Could not load custom model: {str(e)}. Using advanced sentiment analysis.")
        return None, None

# Initialize components
@st.cache_resource
def initialize_components():
    scraper = IMDBScraper()
    model, tokenizer = load_sentiment_model()
    return scraper, model, tokenizer

def preprocess_text_for_model(text, tokenizer, max_length=100):
    """Preprocess text for the trained model"""
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        # Clean text
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
        text = text.lower().strip()
        
        # Tokenize and pad
        if tokenizer:
            sequences = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
            return padded
        else:
            return None
    except ImportError:
        return None

def predict_sentiment(text, model, tokenizer):
    """Predict sentiment using the loaded model"""
    if model is None or tokenizer is None:
        # Fallback to basic sentiment analysis
        return analyze_sentiment_basic(text)
    
    try:
        processed_text = preprocess_text_for_model(text, tokenizer)
        if processed_text is None:
            return analyze_sentiment_basic(text)
        
        prediction = model.predict(processed_text, verbose=0)
        
        # Assuming binary classification (0=negative, 1=positive)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            sentiment = "POSITIVE"
            confidence_score = confidence
        else:
            sentiment = "NEGATIVE" 
            confidence_score = 1 - confidence
        
        return sentiment, confidence_score
        
    except Exception as e:
        st.warning(f"Model prediction failed: {str(e)}. Using fallback analysis.")
        return analyze_sentiment_basic(text)

def analyze_sentiment_basic(text):
    """Advanced keyword-based sentiment analysis"""
    
    # Comprehensive sentiment dictionaries
    positive_words = {
        'excellent': 3, 'outstanding': 3, 'phenomenal': 3, 'masterpiece': 3, 'brilliant': 3,
        'amazing': 2.5, 'fantastic': 2.5, 'wonderful': 2.5, 'superb': 2.5, 'magnificent': 2.5,
        'great': 2, 'good': 2, 'impressive': 2, 'remarkable': 2, 'exceptional': 2,
        'love': 2, 'perfect': 3, 'beautiful': 2, 'stunning': 2.5, 'incredible': 2.5,
        'entertaining': 1.5, 'enjoyable': 1.5, 'fun': 1.5, 'interesting': 1.5, 'engaging': 2,
        'recommend': 2, 'worth': 1.5, 'liked': 1.5, 'solid': 1.5, 'decent': 1.2
    }
    
    negative_words = {
        'terrible': 3, 'horrible': 3, 'awful': 3, 'disaster': 3, 'garbage': 3,
        'worst': 2.5, 'hate': 2.5, 'disgusting': 2.5, 'pathetic': 2.5, 'ridiculous': 2.5,
        'bad': 2, 'poor': 2, 'disappointing': 2, 'boring': 2, 'stupid': 2,
        'waste': 2.5, 'pointless': 2, 'annoying': 1.5, 'dull': 1.5, 'bland': 1.5,
        'confusing': 1.5, 'messy': 1.5, 'failed': 2, 'lacks': 1.5, 'weak': 1.5,
        'avoid': 2, 'skip': 1.5, 'regret': 2, 'disappointed': 2
    }
    
    # Negation words that flip sentiment
    negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', 'none', "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't"]
    
    # Intensifiers
    intensifiers = {'very': 1.5, 'extremely': 2, 'really': 1.3, 'absolutely': 1.8, 'completely': 1.6, 'totally': 1.4, 'quite': 1.2, 'rather': 1.1}
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    positive_score = 0
    negative_score = 0
    total_sentiment_words = 0
    
    for i, word in enumerate(words):
        # Check for negation in previous 3 words
        negated = False
        for j in range(max(0, i-3), i):
            if words[j] in negation_words:
                negated = True
                break
        
        # Check for intensifiers in previous 2 words
        intensity = 1.0
        for j in range(max(0, i-2), i):
            if words[j] in intensifiers:
                intensity = intensifiers[words[j]]
                break
        
        # Calculate sentiment
        if word in positive_words:
            score = positive_words[word] * intensity
            if negated:
                negative_score += score
            else:
                positive_score += score
            total_sentiment_words += 1
            
        elif word in negative_words:
            score = negative_words[word] * intensity
            if negated:
                positive_score += score
            else:
                negative_score += score
            total_sentiment_words += 1
    
    # Calculate confidence based on sentiment word density
    sentiment_density = total_sentiment_words / max(len(words), 1)
    base_confidence = min(0.9, 0.4 + sentiment_density * 2)
    
    # Determine sentiment
    if positive_score > negative_score:
        sentiment = "POSITIVE"
        strength = positive_score - negative_score
        confidence = min(0.95, base_confidence + (strength * 0.1))
    elif negative_score > positive_score:
        sentiment = "NEGATIVE"
        strength = negative_score - positive_score
        confidence = min(0.95, base_confidence + (strength * 0.1))
    else:
        sentiment = "NEUTRAL"
        confidence = 0.5
    
    return sentiment, confidence

def batch_analyze_sentiment(reviews, model, tokenizer):
    """Analyze sentiment for multiple reviews"""
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, review in enumerate(reviews):
        if i % 20 == 0:
            progress = i / len(reviews)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing review {i+1}/{len(reviews)}...")
        
        sentiment, confidence = predict_sentiment(review['text'], model, tokenizer)
        results.append((sentiment, confidence))
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return results

def main():
    # Initialize components
    scraper, model, tokenizer = initialize_components()
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze movie reviews using your custom-trained sentiment analysis model**")
    
    # Model status
    col1, col2 = st.columns([3, 1])
    with col1:
        if model is not None:
            st.success("✅ Custom model loaded and ready")
        else:
            st.error("❌ Model not loaded - upload 'best_model.h5' to use custom analysis")
    
    with col2:
        if st.button("🔄 Reload Model"):
            st.cache_resource.clear()
            st.rerun()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page:",
        ["🔍 Movie Search", "📊 Recent Analyses", "ℹ️ About"]
    )
    
    # Main content
    if page == "🔍 Movie Search":
        show_movie_search(scraper, model, tokenizer)
    elif page == "📊 Recent Analyses":
        show_recent_analyses()
    else:
        show_about_page()

def show_movie_search(scraper, model, tokenizer):
    st.header("🔍 Movie Search & Analysis")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input(
            "Enter movie title:",
            placeholder="e.g., The Dark Knight, Inception, Avengers"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        search_button = st.button("🔍 Search & Analyze", type="primary")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            num_reviews = st.slider("Number of reviews to analyze", 50, 500, 250, 25)
        with col2:
            min_confidence = st.slider("Minimum confidence threshold", 0.5, 0.95, 0.7, 0.05)
    
    if search_button and search_query:
        analyze_movie(search_query, scraper, model, tokenizer, num_reviews, min_confidence)

def analyze_movie(movie_title, scraper, model, tokenizer, num_reviews, min_confidence):
    st.markdown(f"## 🎬 Analysis: {movie_title}")
    
    # Cache key for this analysis
    cache_key = f"analysis_{movie_title.lower().replace(' ', '_')}_{num_reviews}"
    
    # Check if we have cached results
    if cache_key in st.session_state:
        st.info("📋 Using cached results. Click 'Search & Analyze' again to refresh.")
        display_results(st.session_state[cache_key], min_confidence)
        return
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        overall_progress = st.progress(0)
        status_text = st.empty()
    
    try:
        # Step 1: Get movie information
        status_text.text("🎬 Searching for movie...")
        overall_progress.progress(10)
        
        movie_info = scraper.get_movie_info(movie_title)
        if not movie_info:
            st.error(f"❌ Movie '{movie_title}' not found. Please check the spelling and try again.")
            return
        
        # Display movie info
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Title", movie_info['title'])
            with col2:
                st.metric("Year", movie_info.get('year', 'N/A'))
            with col3:
                st.metric("IMDB Rating", movie_info.get('rating', 'N/A'))
            with col4:
                st.metric("Director", movie_info.get('director', 'N/A'))
        
        overall_progress.progress(25)
        
        # Step 2: Scrape reviews
        status_text.text(f"📝 Scraping {num_reviews} reviews...")
        
        reviews = scraper.scrape_reviews(movie_info['imdb_id'], num_reviews)
        
        if len(reviews) < 10:
            st.error(f"❌ Only found {len(reviews)} reviews. Need at least 10 for reliable analysis.")
            
            # Show debug information
            with st.expander("🔍 Debug Information"):
                debug_info = scraper.debug_reviews_page(movie_info['imdb_id'])
                st.json(debug_info)
            return
        
        overall_progress.progress(50)
        status_text.text(f"✅ Found {len(reviews)} reviews. Analyzing sentiment...")
        
        # Step 3: Sentiment analysis
        sentiment_results = batch_analyze_sentiment(reviews, model, tokenizer)
        
        overall_progress.progress(90)
        status_text.text("📊 Compiling results...")
        
        # Compile final results
        analysis_results = {
            'movie_info': movie_info,
            'reviews': reviews,
            'sentiment_results': sentiment_results,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_reviews': len(reviews),
            'model_used': 'Custom Trained Model' if model else 'Basic Fallback'
        }
        
        # Cache results
        st.session_state[cache_key] = analysis_results
        
        overall_progress.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        
        # Clear progress
        progress_container.empty()
        
        # Display results
        display_results(analysis_results, min_confidence)
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

def display_results(results, min_confidence):
    movie_info = results['movie_info']
    reviews = results['reviews']
    sentiment_results = results['sentiment_results']
    
    # Filter by confidence
    filtered_results = [(s, c) for s, c in sentiment_results if c >= min_confidence]
    if len(filtered_results) < len(sentiment_results):
        st.info(f"📊 Filtered to {len(filtered_results)} reviews with confidence ≥ {min_confidence:.0%}")
        sentiment_results = filtered_results
    
    # Calculate statistics
    sentiments = [result[0] for result in sentiment_results]
    confidences = [result[1] for result in sentiment_results]
    
    positive_count = sentiments.count('POSITIVE')
    negative_count = sentiments.count('NEGATIVE')
    neutral_count = sentiments.count('NEUTRAL')
    total_count = len(sentiments)
    
    if total_count == 0:
        st.warning("❌ No reviews meet the confidence threshold. Try lowering the minimum confidence.")
        return
    
    positive_pct = (positive_count / total_count) * 100
    negative_pct = (negative_count / total_count) * 100
    avg_confidence = np.mean(confidences)
    
    # Overall verdict
    if positive_pct > negative_pct + 15:
        verdict = "🎉 HIGHLY POSITIVE"
        verdict_class = "verdict-positive"
    elif positive_pct > negative_pct + 5:
        verdict = "👍 POSITIVE"
        verdict_class = "verdict-positive"
    elif negative_pct > positive_pct + 15:
        verdict = "👎 HIGHLY NEGATIVE"
        verdict_class = "verdict-negative"
    elif negative_pct > positive_pct + 5:
        verdict = "😞 NEGATIVE"
        verdict_class = "verdict-negative"
    else:
        verdict = "😐 MIXED REVIEWS"
        verdict_class = "verdict-mixed"
    
    # Display verdict prominently
    st.markdown(f'<div class="{verdict_class}">{verdict}</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Positive Reviews",
            f"{positive_count:,}",
            f"{positive_pct:.1f}%"
        )
    
    with col2:
        st.metric(
            "Negative Reviews",
            f"{negative_count:,}",
            f"{negative_pct:.1f}%"
        )
    
    with col3:
        st.metric(
            "Average Confidence",
            f"{avg_confidence:.1%}",
            f"Total: {total_count:,}"
        )
    
    with col4:
        recommendation = "👍 RECOMMENDED" if positive_pct > 60 else "👎 NOT RECOMMENDED" if negative_pct > 60 else "🤔 DEPENDS ON TASTE"
        st.metric("Recommendation", recommendation)
    
    # Visualizations
    st.markdown("---")
    st.subheader("📊 Analysis Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution
        fig_pie = px.pie(
            values=[positive_count, negative_count, neutral_count] if neutral_count > 0 else [positive_count, negative_count],
            names=['Positive', 'Negative', 'Neutral'] if neutral_count > 0 else ['Positive', 'Negative'],
            title="Sentiment Distribution",
            color_discrete_map={
                'Positive': '#28a745',
                'Negative': '#dc3545',
                'Neutral': '#ffc107'
            }
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence distribution
        fig_hist = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Number of Reviews'},
            color_discrete_sequence=['#007bff']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Sample reviews
    st.markdown("---")
    st.subheader("📝 Sample Reviews")
    
    # Combine reviews with sentiment data
    reviews_with_sentiment = []
    for i, (sentiment, confidence) in enumerate(sentiment_results):
        if i < len(reviews):
            review = reviews[i].copy()
            review['sentiment'] = sentiment
            review['confidence'] = confidence
            reviews_with_sentiment.append(review)
    
    # Filter and sort
    positive_reviews = [r for r in reviews_with_sentiment if r['sentiment'] == 'POSITIVE']
    negative_reviews = [r for r in reviews_with_sentiment if r['sentiment'] == 'NEGATIVE']
    
    positive_reviews.sort(key=lambda x: x['confidence'], reverse=True)
    negative_reviews.sort(key=lambda x: x['confidence'], reverse=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 😊 Top Positive Reviews")
        for i, review in enumerate(positive_reviews[:3]):
            with st.expander(f"Positive Review {i+1} (Confidence: {review['confidence']:.1%})"):
                st.write(review['text'][:400] + "..." if len(review['text']) > 400 else review['text'])
                st.caption(f"Author: {review.get('author', 'Anonymous')} | Rating: {review.get('rating', 'N/A')}")
    
    with col2:
        st.markdown("#### 😞 Top Negative Reviews")
        for i, review in enumerate(negative_reviews[:3]):
            with st.expander(f"Negative Review {i+1} (Confidence: {review['confidence']:.1%})"):
                st.write(review['text'][:400] + "..." if len(review['text']) > 400 else review['text'])
                st.caption(f"Author: {review.get('author', 'Anonymous')} | Rating: {review.get('rating', 'N/A')}")
    
    # Analysis metadata
    st.markdown("---")
    st.caption(f"Analysis completed on {results['analysis_date']} using {results['model_used']}")

def show_recent_analyses():
    st.header("📊 Recent Analyses")
    
    # Get all cached analyses from session state
    analyses = {k: v for k, v in st.session_state.items() if k.startswith('analysis_')}
    
    if not analyses:
        st.info("No recent analyses found. Search for movies to see results here.")
        return
    
    # Display analyses summary
    st.subheader("Analysis Summary")
    
    analysis_data = []
    for key, result in analyses.items():
        movie_info = result['movie_info']
        sentiment_results = result['sentiment_results']
        
        sentiments = [r[0] for r in sentiment_results]
        positive_pct = (sentiments.count('POSITIVE') / len(sentiments)) * 100
        
        analysis_data.append({
            'Movie': movie_info['title'],
            'Year': movie_info.get('year', 'N/A'),
            'Total Reviews': result['total_reviews'],
            'Positive %': positive_pct,
            'Analysis Date': result['analysis_date'],
            'IMDB Rating': movie_info.get('rating', 'N/A')
        })
    
    df = pd.DataFrame(analysis_data)
    df = df.sort_values('Analysis Date', ascending=False)
    
    # Display as table
    st.dataframe(df, use_container_width=True)
    
    # Comparison chart
    if len(df) > 1:
        st.subheader("Comparison Chart")
        fig = px.scatter(
            df,
            x='Total Reviews',
            y='Positive %',
            size='Total Reviews',
            hover_name='Movie',
            title="Movie Sentiment Comparison",
            labels={'Positive %': 'Positive Review %'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Clear cache option
    if st.button("🗑️ Clear All Analyses"):
        for key in list(st.session_state.keys()):
            if key.startswith('analysis_'):
                del st.session_state[key]
        st.success("All analyses cleared!")
        st.rerun()

def show_about_page():
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ### 🎬 Movie Review Sentiment Analysis
    
    This application combines web scraping and machine learning to analyze movie reviews from IMDB.
    
    **Features:**
    - **Custom Model Integration**: Uses your pre-trained sentiment analysis model (`best_model.h5`)
    - **IMDB Scraping**: Extracts up to 500 reviews per movie with robust scraping techniques
    - **Real-time Analysis**: Processes reviews and provides immediate sentiment insights
    - **Interactive Visualizations**: Charts and graphs to understand sentiment distribution
    - **Confidence Filtering**: Filter results by prediction confidence levels
    
    **How it works:**
    1. **Search**: Enter a movie title to find it on IMDB
    2. **Scrape**: The app extracts reviews from multiple pages
    3. **Analyze**: Your trained model predicts sentiment for each review
    4. **Visualize**: Results are displayed with charts and sample reviews
    
    **Model Requirements:**
    - Upload your trained model as `best_model.h5`
    - Optionally include `tokenizer.pickle` for better preprocessing
    - Model should output binary classification (0=negative, 1=positive)
    
    **Technical Details:**
    - Built with Streamlit for the web interface
    - Uses BeautifulSoup for web scraping with multiple fallback strategies
    - TensorFlow/Keras for model inference
    - Plotly for interactive visualizations
    """)
    
    st.markdown("---")
    st.markdown("**Made with ❤️ using Streamlit and TensorFlow**")

if __name__ == "__main__":
    main()
