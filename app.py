import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import StringIO
import time

# --- CONFIG ---
st.set_page_config(
    page_title="AI Redirect Mapper", 
    page_icon="🔄", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Instructions card */
    .instructions-card {
        background-color: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Stats cards */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: bold;
        display: block;
    }
    
    .stats-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Result styling */
    .result-header {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
        color: var(--text-color);
    }
    
    .footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
    }
    
    /* Metric styling for light/dark theme compatibility */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    
    [data-testid="metric-container"] > label {
        color: white !important;
    }
    
    [data-testid="metric-container"] > div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<div class="main-header">
    <h1>🔄 AI Redirect Mapping Tool</h1>
    <p>Map old URLs to new URLs using semantic similarity — no API required</p>
</div>
""", unsafe_allow_html=True)

# --- INSTRUCTIONS ---
with st.expander("📖 How to Use This Tool", expanded=False):
    st.markdown("""
    ### 🚀 Quick Start Guide
    
    **Option 1: Try the Demo**
    1. ✅ Check "Use Demo Data" in the sidebar
    2. 🎯 Adjust the similarity threshold (0.5 is a good start)
    3. 📊 View your mapping results instantly
    
    **Option 2: Upload Your Own Data**
    1. 📁 Prepare two CSV files with columns: `url` and `content`
       - **Old URLs CSV**: Your existing URLs and their content/descriptions
       - **New URLs CSV**: Your new URLs and their content/descriptions
    2. 📤 Upload both files using the file uploaders
    3. 🎯 Adjust the similarity threshold based on your needs
    4. 📋 Download the mapping results as CSV
    
    ### 🎯 Understanding Similarity Threshold
    - **0.7-1.0**: Very strict matching (high confidence)
    - **0.5-0.7**: Moderate matching (balanced)
    - **0.3-0.5**: Loose matching (may include false positives)
    
    ### 📝 CSV Format Example
    ```
    url,content
    https://example.com/about-us,About our company mission and values
    https://example.com/products,Our complete product catalog and services
    https://example.com/contact,Contact information and support details
    ```
    
    ### 💡 Pro Tips
    - **Content descriptions should be detailed** for better matching accuracy
    - **Include keywords** that represent the page's main purpose
    - **Use similar language** between old and new content descriptions
    - **Test with demo data first** to understand how the tool works
    """)

# --- Load Model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

with st.spinner("🤖 Loading AI model..."):
    model = load_model()

# --- Sample Data (Embedded) ---
SAMPLE_OLD = """url,content
https://oldsite.com/about-us,About our company history mission statement and core values
https://oldsite.com/products-catalog,Browse our complete product catalog and service offerings
https://oldsite.com/contact-us,Contact us through phone email or contact form
https://oldsite.com/blog/seo-guide,Complete SEO guide and optimization tips for better rankings
https://oldsite.com/our-team,Meet our leadership team and company founders
https://oldsite.com/pricing-plans,View our subscription pricing and service packages
https://oldsite.com/customer-support,Customer service help desk and technical support
https://oldsite.com/job-openings,Current career opportunities and job listings
https://oldsite.com/testimonials,Customer reviews testimonials and success stories
https://oldsite.com/news-updates,Latest company news press releases and updates"""

SAMPLE_NEW = """url,content
https://newsite.com/about,Learn about our company mission vision and history
https://newsite.com/products,Explore our product range and service solutions
https://newsite.com/contact,Get in touch with our team via contact form or phone
https://newsite.com/resources/seo-tips,SEO best practices and search engine optimization strategies
https://newsite.com/team,Our leadership team founders and key personnel
https://newsite.com/pricing,Pricing plans subscription options and packages
https://newsite.com/support,Customer support technical help and assistance
https://newsite.com/careers,Join our team current job vacancies and opportunities
https://newsite.com/reviews,Client testimonials customer feedback and case studies
https://newsite.com/blog/company-news,Company announcements news and press coverage"""

# --- Load Demo Data Function ---
def load_demo_data():
    old_df = pd.read_csv(StringIO(SAMPLE_OLD))
    new_df = pd.read_csv(StringIO(SAMPLE_NEW))
    return old_df, new_df

# --- Sidebar: Demo or Upload ---
st.sidebar.markdown("### 📊 Data Input Options")
use_demo = st.sidebar.checkbox("✨ Use Demo Data (Sample CSVs)", help="Try the tool with sample data first")

if use_demo:
    st.sidebar.success("🎯 Demo mode activated!")
    st.sidebar.markdown("**Sample data includes:**\n- 10 old URLs\n- 10 new URLs\n- Realistic website migration scenario\n- Expected: 90%+ match rate")

# --- Main Content ---
col1, col2 = st.columns([1, 1])

# --- File Upload or Demo ---
if use_demo:
    old_df, new_df = load_demo_data()
    
    with col1:
        st.success("✅ Old URLs loaded (Demo)")
        st.info(f"📊 {len(old_df)} URLs loaded")
        with st.expander("Preview Old URLs"):
            st.dataframe(old_df, use_container_width=True)
    
    with col2:
        st.success("✅ New URLs loaded (Demo)")
        st.info(f"📊 {len(new_df)} URLs loaded")
        with st.expander("Preview New URLs"):
            st.dataframe(new_df, use_container_width=True)
else:
    with col1:
        st.markdown("#### 📁 Upload Old URLs CSV")
        old_file = st.file_uploader(
            "Choose old URLs file", 
            type="csv", 
            help="CSV with columns: url, content",
            key="old_file"
        )
        
    with col2:
        st.markdown("#### 📁 Upload New URLs CSV")
        new_file = st.file_uploader(
            "Choose new URLs file", 
            type="csv", 
            help="CSV with columns: url, content",
            key="new_file"
        )

    old_df, new_df = None, None
    
    if old_file and new_file:
        try:
            old_df = pd.read_csv(old_file)
            new_df = pd.read_csv(new_file)
            
            # Validate columns
            required_cols = ['url', 'content']
            if not all(col in old_df.columns for col in required_cols):
                st.error("❌ Old URLs CSV must have 'url' and 'content' columns")
                old_df = None
            elif not all(col in new_df.columns for col in required_cols):
                st.error("❌ New URLs CSV must have 'url' and 'content' columns")
                new_df = None
            else:
                with col1:
                    st.success("✅ Old URLs loaded successfully")
                    st.info(f"📊 {len(old_df)} URLs loaded")
                    
                with col2:
                    st.success("✅ New URLs loaded successfully")
                    st.info(f"📊 {len(new_df)} URLs loaded")
                    
        except Exception as e:
            st.error(f"❌ Error loading files: {str(e)}")

# --- Configuration ---
if (use_demo or (old_df is not None and new_df is not None)):
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        threshold = st.slider(
            "🎯 Similarity Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.01,
            help="Higher values = stricter matching. 0.5 is recommended for most cases."
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_top_n = st.selectbox("Show top N matches per URL", [1, 2, 3, 5], index=0, help="Show multiple potential matches")
            exclude_low_similarity = st.checkbox("Auto-exclude matches below 0.3", value=True, help="Skip very poor matches")
    
    with col2:
        st.metric("Old URLs", len(old_df), help="Number of URLs to map from")
    
    with col3:
        st.metric("New URLs", len(new_df), help="Number of URLs to map to")

# --- Mapping Logic ---
if (use_demo or (old_df is not None and new_df is not None)) and st.button("🚀 Start Mapping", type="primary", use_container_width=True):
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Generate embeddings for old URLs
        status_text.text("🔄 Processing old URLs...")
        progress_bar.progress(25)
        old_embeddings = model.encode(old_df['content'].astype(str).tolist(), show_progress_bar=False)
        
        # Step 2: Generate embeddings for new URLs
        status_text.text("🔄 Processing new URLs...")
        progress_bar.progress(50)
        new_embeddings = model.encode(new_df['content'].astype(str).tolist(), show_progress_bar=False)
        
        # Step 3: Compute similarity
        status_text.text("🧮 Computing similarities...")
        progress_bar.progress(75)
        sim_matrix = cosine_similarity(old_embeddings, new_embeddings)
        
        # Step 4: Find best matches
        status_text.text("🎯 Finding best matches...")
        progress_bar.progress(90)
        
        results = []
        good_matches = 0
        
        for i, old_url in enumerate(old_df['url']):
            # Get top N matches
            top_indices = np.argsort(sim_matrix[i])[::-1][:show_top_n]
            
            for rank, idx in enumerate(top_indices):
                score = sim_matrix[i][idx]
                
                # Skip very low similarity if enabled
                if exclude_low_similarity and score < 0.3:
                    continue
                
                if score >= threshold:
                    status = "✅ Good Match"
                    if rank == 0:  # Only count first match for stats
                        good_matches += 1
                elif score >= 0.3:
                    status = "⚠️ Low Confidence"
                else:
                    status = "❌ Poor Match"
                
                # Add confidence level
                if score >= 0.8:
                    confidence = "🔥 Excellent"
                elif score >= 0.6:
                    confidence = "👍 Good"
                elif score >= 0.4:
                    confidence = "🤔 Fair"
                else:
                    confidence = "👎 Poor"
                
                match_type = "🥇 Primary" if rank == 0 else f"🥈 Alternative #{rank+1}"
                
                results.append({
                    "Old URL": old_url,
                    "New URL": new_df['url'][idx] if score >= threshold else "No good match found",
                    "Similarity": f"{score:.3f}",
                    "Status": status,
                    "Confidence": confidence,
                    "Match Type": match_type,
                    "Old Content": old_df['content'][i][:80] + "..." if len(str(old_df['content'][i])) > 80 else old_df['content'][i],
                    "New Content": new_df['content'][idx][:80] + "..." if len(str(new_df['content'][idx])) > 80 else new_df['content'][idx]
                })
        
        progress_bar.progress(100)
        status_text.text("✅ Mapping completed!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Results
        result_df = pd.DataFrame(results)
        
        st.markdown(f"""
        <div class="result-header">
            <h3>🎉 Mapping Results Complete!</h3>
            <p>{good_matches} out of {len(old_df)} URLs found good matches</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total URLs", len(old_df))
        with col2:
            st.metric("Good Matches", good_matches, f"{(good_matches/len(old_df)*100):.1f}%")
        with col3:
            st.metric("Low Confidence", len(old_df) - good_matches)
        with col4:
            unique_urls = len(result_df['Old URL'].unique())
            avg_similarity = np.mean([float(x) for x in result_df['Similarity']])
            st.metric("Avg Similarity", f"{avg_similarity:.3f}", f"{unique_urls} unique URLs")
        
        # Results table
        st.markdown("### 📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_all = st.checkbox("Show all results", value=True)
        with col2:
            if not show_all:
                filter_type = st.selectbox("Filter by", ["Good matches only", "By confidence level"])
            else:
                filter_type = None
        with col3:
            if not show_all and filter_type == "By confidence level":
                conf_filter = st.selectbox("Confidence", ["🔥 Excellent", "👍 Good", "🤔 Fair", "👎 Poor"])
        
        # Apply filters
        if show_all:
            display_df = result_df
        elif filter_type == "Good matches only":
            display_df = result_df[result_df['Status'] == '✅ Good Match']
        else:
            display_df = result_df[result_df['Confidence'] == conf_filter]
        
        # Quick stats
        if len(display_df) < len(result_df):
            st.info(f"Showing {len(display_df)} of {len(result_df)} results")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "Similarity": st.column_config.ProgressColumn(
                    "Similarity Score",
                    help="Semantic similarity score",
                    min_value=0,
                    max_value=1,
                    format="%.3f"
                ),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                "Match Type": st.column_config.TextColumn("Type", width="small"),
                "Old URL": st.column_config.LinkColumn("Old URL"),
                "New URL": st.column_config.LinkColumn("New URL")
            }
        )
        
        # Download options
        st.markdown("### 📥 Download Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Full results
            csv_full = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Full Results",
                csv_full,
                "redirect_mapping_full.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Only good matches
            good_matches_df = result_df[result_df['Status'] == '✅ Good Match']
            csv_good = good_matches_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "✅ Download Good Matches Only",
                csv_good,
                "redirect_mapping_good_matches.csv",
                "text/csv",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        progress_bar.empty()
        status_text.empty()

elif not use_demo and (old_df is None or new_df is None):
    st.info("👆 Please upload both CSV files or enable demo mode to get started.")

# --- Footer ---
st.markdown("""
<div class="footer">
    <p>🛠️ Built with ❤️ by <strong><a href="https://www.linkedin.com/in/amal-alexander-305780131/" target="_blank">Amal Alexander</a></strong></p>
    <p>Powered by Sentence Transformers & Streamlit</p>
</div>
""", unsafe_allow_html=True)