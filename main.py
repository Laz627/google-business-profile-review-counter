import time
import streamlit as st
import pandas as pd
import numpy as np
from client import RestClient
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Advanced AI Review Analyzer")

# --- CORE ANALYSIS FUNCTIONS ---

def get_embeddings(api_key, texts, model="text-embedding-3-small"):
    """Generates embeddings for a list of texts in a single batch call."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}")
        return None

def summarize_cluster(api_key, reviews_sample, model="gpt-4o-mini"):
    """Summarizes a sample of reviews from a cluster into a short theme name."""
    reviews_text = "\n".join("- " + r for r in reviews_sample)
    prompt = f"""
    Analyze these customer reviews. What is the single most dominant theme?
    Respond with a short, descriptive theme name of 4-6 words.
    Example responses: 'Fast and friendly customer service', 'High prices and unexpected fees', 'Long shipping and delivery times'.

    Reviews:
    {reviews_text}
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at identifying themes in text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Unnamed Theme"

def analyze_reviews_with_embeddings(api_key, business_name, reviews_df, num_clusters=7, model="gpt-4o-mini"):
    """
    Performs an advanced review analysis using embeddings, clustering, and recency analysis.
    This function is designed to be run concurrently for each business.
    """
    if reviews_df.empty or len(reviews_df) < num_clusters:
        return business_name, "Not enough reviews available for a detailed analysis."

    # --- Stage 1: Embedding Generation ---
    review_texts = reviews_df['Review Body'].tolist()
    embeddings = get_embeddings(api_key, review_texts)
    if not embeddings:
        return business_name, "Could not generate embeddings for reviews."

    # --- Stage 2: Clustering ---
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    reviews_df['cluster'] = kmeans.fit_predict(np.array(embeddings))
    reviews_df['rating_numeric'] = pd.to_numeric(reviews_df['Rating'], errors='coerce')

    # --- Stage 3: Concurrent Theme Summarization ---
    cluster_themes = {}
    with ThreadPoolExecutor(max_workers=num_clusters) as executor:
        future_to_cluster = {
            executor.submit(
                summarize_cluster, api_key,
                reviews_df[reviews_df['cluster'] == i]['Review Body'].head(10).tolist() # Sample up to 10 reviews
            ): i
            for i in range(num_clusters)
        }
        for future in as_completed(future_to_cluster):
            cluster_id = future_to_cluster[future]
            theme_name = future.result()
            cluster_themes[cluster_id] = theme_name

    # --- Stage 4: Recency Analysis ---
    reviews_df['Timestamp'] = pd.to_datetime(reviews_df['Timestamp'], errors='coerce')
    ninety_days_ago = pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_reviews = reviews_df[reviews_df['Timestamp'] >= ninety_days_ago]['Review Body'].tolist()
    older_reviews = reviews_df[reviews_df['Timestamp'] < ninety_days_ago]['Review Body'].tolist()
    
    recency_summary = ""
    if recent_reviews:
        recency_prompt = f"Summarize the key topics from these RECENT customer reviews (from the last 90 days) for '{business_name}'.\n\n" + "\n".join("- " + r for r in recent_reviews[:20])
        # This can be another GPT call, but for simplicity we will include it in the final prompt.

    # --- Stage 5: Final Synthesis ---
    final_report_data = []
    for i in range(num_clusters):
        cluster_data = reviews_df[reviews_df['cluster'] == i]
        avg_rating = cluster_data['rating_numeric'].mean()
        sentiment = "Positive" if avg_rating >= 3.5 else ("Negative" if avg_rating < 2.5 else "Mixed")
        final_report_data.append(
            f"Theme: '{cluster_themes[i]}', "
            f"Sentiment: {sentiment} (Avg Rating: {avg_rating:.2f}), "
            f"Prevalence: {len(cluster_data)/len(reviews_df):.1%}"
        )

    final_prompt = f"""
    You are a business intelligence analyst creating a report for the executive team of '{business_name}'.
    You have analyzed customer reviews using an AI clustering model. Your task is to synthesize the following structured data into a clear, actionable report in Markdown format.

    **Extracted Review Themes:**
    {chr(10).join(f'- {item}' for item in final_report_data)}

    **Recent Feedback Summary (last 90 days):**
    {"- " + chr(10).join(recent_reviews[:20]) if recent_reviews else "No recent reviews with text."}

    **Instructions:**
    Based *only* on the data provided, generate the following:
    1.  **Executive Summary:** A 2-3 sentence overview of the key findings.
    2.  **Dominant Positive Themes (Pros):** A bulleted list of the main positive themes.
    3.  **Key Areas for Improvement (Cons):** A bulleted list of the main negative themes.
    4.  **Recency Analysis:** A short paragraph describing what recent customers are talking about and if it differs from older feedback.
    5.  **Strategic Recommendation:** A concluding sentence on what the business should focus on.
    """

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.4,
        )
        return business_name, response.choices[0].message.content
    except Exception as e:
        return business_name, f"Failed to generate final report: {e}"

# --- STREAMLIT UI AND MAIN WORKFLOW ---

def main():
    st.title("Scalable AI Review Analyzer with Embeddings")

    with st.expander("ℹ️ How This Tool Works", expanded=True):
        st.markdown("""
        This tool performs a deep analysis of Google Reviews for any business using a scalable AI pipeline, ideal for large numbers of reviews.

        **Here's the process:**
        1.  **Fetch Reviews:** It retrieves reviews for your specified businesses using the DataForSEO API.
        2.  **AI Embeddings:** Each review is converted into a numerical representation (an "embedding") by OpenAI's AI, capturing its core meaning.
        3.  **Clustering:** An algorithm groups reviews with similar meanings into distinct **thematic clusters**. This automatically discovers the main topics customers are talking about (e.g., service speed, price, quality).
        4.  **AI Summarization:** A generative AI model (`gpt-4o-mini`) analyzes each cluster to assign a human-readable theme name and summarizes recent vs. older feedback.
        5.  **Synthesized Report:** The final output provides a strategic overview, including pros, cons, and emerging trends from recent reviews.
        
        This advanced method is faster and more accurate for large datasets than summarizing all reviews at once.
        """)
    
    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataForSEO API Login", type="password")
        api_password = st.text_input("DataForSEO API Password", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        st.header("Analysis Settings")
        depth = st.number_input("Reviews to Fetch per Profile", 100, 10000, 500, 100)
        num_clusters = st.slider("Number of Themes to Extract (Clusters)", 4, 10, 7)

    # --- Main App Body ---
    st.markdown("### 1. Enter Google Business Profiles")
    profiles_input = st.text_area("One per line: **Business Name, Location**", height=120, placeholder="Pella Windows and Doors, Chesterfield, MO, United States\nExample Restaurant, New York, NY")
    
    if st.button("🚀 Run Advanced Analysis", use_container_width=True):
        if not all([api_login, api_password, openai_api_key, profiles_input.strip()]):
            st.error("Please provide all API credentials and at least one GBP profile.")
            return
        
        # --- Data Fetching (Reusing optimized function) ---
        from main import get_review_counts_optimized # Assuming this is in main.py
        tasks_list = [
            {"keyword": ", ".join(parts[:-1]), "location_name": parts[-1]} if len(parts) > 1 else {"keyword": parts[0], "location_name": "United States"}
            for line in profiles_input.strip().splitlines() if (parts := [p.strip() for p in line.split(",")])
        ]
        
        with st.spinner("Fetching reviews using efficient batch polling..."):
            # NOTE: Assuming get_review_counts_optimized is available from the previous step's code.
            # You would integrate that function here. For this example, we'll proceed with dummy logic.
            # results = get_review_counts_optimized(api_login, api_password, tasks_list, depth)
            pass # Placeholder
        
        # This section is for a self-contained example. Replace with your live fetching.
        # --- Start of Placeholder Data ---
        # In a real run, this DataFrame would be generated by your DataForSEO calls.
        from main import get_detailed_reviews_dataframe
        # results = get_review_counts_optimized(api_login, api_password, tasks_list, depth)
        # all_reviews_df = get_detailed_reviews_dataframe(results)
        # --- End of Placeholder Data ---
        
        # The following line should be your actual fetched data
        # all_reviews_df = get_detailed_reviews_dataframe(results)
        st.info("Bypassing live fetch for demonstration. Using sample data.")
        all_reviews_df = pd.DataFrame({ # Sample data
            'Business': ['Pella Windows and Doors'] * 5 + ['Example Restaurant'] * 5,
            'Location': ['Chesterfield, MO'] * 10,
            'Rating': [5, 1, 4, 5, 2, 5, 5, 2, 1, 4],
            'Review Body': [
                "The installation team was professional and clean.", "The salesperson was an hour late to our appointment.",
                "Good quality windows, but the price was higher than expected.", "Absolutely love our new door. The whole process was smooth.",
                "Communication was poor, had to call multiple times for an update.", "The food here is absolutely divine! Best pasta in the city.",
                "Service was impeccable. Our waiter, John, was fantastic.", "The music was way too loud and the table was sticky.",
                "My order was completely wrong and the manager was rude about it.", "A bit pricey but you get what you pay for. The quality is unmatched."
            ],
            'Timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=10, freq='-30D'))
        })

        if all_reviews_df.empty:
            st.error("No reviews with text content were found to analyze.")
            return

        st.markdown("---")
        st.header("🤖 AI Analysis Reports")
        
        # --- Concurrent Analysis ---
        business_dfs = {name: group for name, group in all_reviews_df.groupby('Business')}
        
        with st.spinner(f"Analyzing {len(business_dfs)} businesses concurrently with the embedding pipeline..."):
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_business = {
                    executor.submit(analyze_reviews_with_embeddings, openai_api_key, name, df.copy(), num_clusters): name
                    for name, df in business_dfs.items()
                }
                placeholders = {name: st.empty() for name in business_dfs.keys()}
                for future in as_completed(future_to_business):
                    business_name, report = future.result()
                    with placeholders[business_name].container():
                        st.subheader(f"Analysis for: {business_name}")
                        st.markdown(report)
                        st.markdown("---")
        
        st.header("📄 Detailed Review Data Used for Analysis")
        st.dataframe(all_reviews_df)

if __name__ == "__main__":
    main()
