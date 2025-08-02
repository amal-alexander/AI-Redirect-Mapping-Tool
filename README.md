# 🔄 AI Redirect Mapping Tool

Map old URLs to new URLs using semantic similarity — no API required.

![AI Redirect Mapper Banner](https://img.shields.io/badge/Streamlit-Powered-764ba2?logo=streamlit&style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-667eea?style=for-the-badge)

## 🚀 Overview

**AI Redirect Mapping Tool** helps website owners, SEOs, and developers map old URLs to new ones by analyzing the semantic similarity between their content or descriptions. No API or login required—just upload your CSVs and instantly generate redirect mapping suggestions!

- **Fast and Easy**: Upload, configure, and download results—all in one page.
- **AI-Powered**: Uses [Sentence Transformers](https://www.sbert.net/) for semantic matching.
- **No Coding Needed**: 100% Streamlit UI, no scripts or API keys required.
- **Flexible**: Use demo data or your own CSVs.

---

## 🖥️ Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-redirect-mapping-tool-5nhju5qfztxfq5n9gpnbhl.streamlit.app/)

---

## 📖 How to Use This Tool

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

```csv
url,content
https://example.com/about-us,About our company mission and values
https://example.com/products,Our complete product catalog and services
https://example.com/contact,Contact information and support details
```
## Screenshot

<img width="1737" height="901" alt="image" src="https://github.com/user-attachments/assets/5970462e-dc7a-43e0-b09c-b034b9217e5c" />


### 💡 Pro Tips

- **Content descriptions should be detailed** for better matching accuracy
- **Include keywords** that represent the page's main purpose
- **Use similar language** between old and new content descriptions
- **Test with demo data first** to understand how the tool works

---

## 🛠️ Features

- **Semantic Matching**: Uses `all-MiniLM-L6-v2` SentenceTransformer model.
- **Custom Threshold**: Tune similarity strictness for your scenario.
- **Multiple Matches**: See top N matches per old URL.
- **Status & Confidence**: Visual markers for match quality.
- **Downloadable Results**: Export full results or only good matches.

---

## 📦 Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/yourusername/ai-redirect-mapper.git
    cd ai-redirect-mapper
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    *Requirements:*
    - `streamlit`
    - `pandas`
    - `sentence-transformers`
    - `scikit-learn`
    - `numpy`

3. **Run the app**

    ```bash
    streamlit run app.py
    ```

---

## ⚙️ Configuration

- **Similarity Threshold**: Adjust using the slider (recommended: 0.5).
- **Top N Matches**: Show multiple matches per old URL.
- **Exclude Low Similarity**: Skip matches below 0.3 for cleaner results.

---

## 📊 Output

- **Status**: ✅ Good Match / ⚠️ Low Confidence / ❌ Poor Match
- **Confidence Level**: 🔥 Excellent, 👍 Good, 🤔 Fair, 👎 Poor
- **Match Type**: 🥇 Primary / 🥈 Alternative
- **Download Options**: Full results or good matches only.

---

## 👤 Credits

Built with ❤️ by [Amal Alexander](https://www.linkedin.com/in/amal-alexander-305780131/)

Powered by [Sentence Transformers](https://www.sbert.net/) & [Streamlit](https://streamlit.io/)

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 🤝 Contributing

Pull requests welcome! Please open issues or suggestions for improvements.
