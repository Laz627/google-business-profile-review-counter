import time
import streamlit as st
import pandas as pd
import numpy as np
from client import RestClient # Ensure client.py is in the repository
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Advanced AI Review Analyzer")

# --- DATA FETCHING FUNCTIONS (FROM PREVIOUS STEPS) ---

def get_review_counts_optimized(api_login, api_password, tasks_list, depth=100):
    """
    Optimized version that uses the 'tasks_ready' endpoint for efficient polling.
    """
    client = RestClient(api_login, api_password)
    
    post_data = {
        idx: {
            "keyword": task["keyword"],
            "location_name": task["location_name"],
            "language_name": "English",
            "depth": depth
        } for idx, task in enumerate(tasks_list)
    }
        
    st.write("**Posting tasks...**")
    post_response = client.post("/v3/business_data/google/reviews/task_post", post_data)
    if post_response.get("status_code") != 20000:
        st.error(f"Error posting tasks: {post_response.get('status_code')} {post_response.get('status_message')}")
        return None
    
    task_ids = [task["id"] for task in post_response.get("tasks", []) if task.get("id")]
    if not task_ids:
        st.error("Could not retrieve task IDs after posting.")
        return None
        
    st.write("Task IDs:", task_ids)
    
    st.write("**Waiting for tasks to complete (using efficient polling)...**")
    completed_results = {}
    completed_task_ids = set()
    start_time = time.time()
    polling_status = st.empty()

    while len(completed_task_ids) < len(task_ids):
        elapsed_time = int(time.time() - start_time)
        polling_status.text(f"**Time elapsed:** {elapsed_time}s. Completed {len(completed_task_ids)}/{len(task_ids)} tasks.")

        ready_response = client.get("/v3/business_data/google/reviews/tasks_ready")
        if ready_response.get("status_code") != 20000:
            st.warning("Could not check task readiness. Will retry...")
            time.sleep(20)
            continue

        ready_tasks = ready_response.get("tasks", [])[0].get("result", [])
        if not ready_tasks:
            time.sleep(20)
            continue
        
        for task_info in ready_tasks:
            task_id = task_info.get("id")
            if task_id and task_id in task_ids and task_id not in completed_task_ids:
                st.write(f"Task {task_id} is ready. Fetching results...")
                result_response = client.get(f"/v3/business_data/google/reviews/task_get/{task_id}")
                if result_response.get("status_code") == 20000:
                    completed_results[task_id] = result_response
                    completed_task_ids.add(task_id)
                else:
                    st.warning(f"Failed to fetch result for completed task {task_id}.")
        
        if len(completed_task_ids) < len(task_ids):
             time.sleep(20)

    polling_status.success(f"All {len(task_ids)} tasks completed in {int(time.time() - start_time)} seconds.")
    return list(completed_results.values())


def get_detailed_reviews_dataframe(results):
    """
    Parses the raw API results into a clean DataFrame for analysis.
    """
    detailed_rows = []
    for res in results:
        if res.get("status_code") == 20000:
            task_result = res.get("tasks", [])[0]
            task_params = task_result.get("data", {})
            business = task_params.get("keyword", "Unknown")
            location = task_params.get("location_name", "Unknown")
            results_array = task_result.get("result", [])
            if results_array:
                items = results_array[0].get("items", [])
                for item in items:
                    review_body = item.get("review_text", "")
                    if review_body:
                        detailed_rows.append({
                            "Business": business, "Location": location,
                            "Rating": item.get("rating", {}).get("value"),
                            "Review Body": review_body,
                            "Timestamp": item.get("timestamp")
                        })
    return pd.DataFrame(detailed_rows) if detailed_rows else pd.DataFrame()


# --- CORE AI ANALYSIS FUNCTIONS ---

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

    # Stage 1: Embedding Generation
    review_texts = reviews_df['Review Body'].tolist()
    embeddings = get_embeddings(api_key, review_texts)
    if not embeddings:
        return business_name, "Could not generate embeddings for reviews."

    # Stage 2: Clustering
    kmeans = KMeans(n_clusters=min(num_clusters, len(review_texts)), random_state=42, n_init='auto')
    reviews_df['cluster'] = kmeans.fit_predict(np.array(embeddings))
    reviews_df['rating_numeric'] = pd.to_numeric(reviews_df['Rating'], errors='coerce')

    # Stage 3: Concurrent Theme Summarization
    cluster_themes = {}
    with ThreadPoolExecutor(max_workers=num_clusters) as executor:
        future_to_cluster = {
            executor.submit(
                summarize_cluster, api_key,
                reviews_df[reviews_df['cluster'] == i]['Review Body'].head(10).tolist()
            ): i
            for i in reviews_df['cluster'].unique()
        }
        for future in as_completed(future_to_cluster):
            cluster_id = future_to_cluster[future]
            cluster_themes[cluster_id] = future.result()

    # Stage 4: Recency Analysis
    reviews_df['Timestamp'] = pd.to_datetime(reviews_df['Timestamp'], errors='coerce')
    ninety_days_ago = pd.Timestamp.now(tz=reviews_df['Timestamp'].dt.tz) - pd.Timedelta(days=90) if reviews_df['Timestamp'].dt.tz else pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_reviews = reviews_df[reviews_df['Timestamp'] >= ninety_days_ago]['Review Body'].tolist()

    # Stage 5: Final Synthesis
    final_report_data = []
    for i, theme in cluster_themes.items():
        cluster_data = reviews_df[reviews_df['cluster'] == i]
        avg_rating = cluster_data['rating_numeric'].mean()
        sentiment = "Positive" if avg_rating >= 3.5 else ("Negative" if avg_rating < 2.5 else "Mixed")
        final_report_data.append(
            f"Theme: '{theme}', Sentiment: {sentiment} (Avg Rating: {avg_rating:.2f}), Prevalence: {len(cluster_data)/len(reviews_df):.1%}"
        )

    final_prompt = f"""
    You are a business intelligence analyst creating a report for '{business_name}'.
    Synthesize the following structured data into a clear, actionable report in Markdown format.

    **Extracted Review Themes:**
    {chr(10).join(f'- {item}' for item in final_report_data)}

    **Recent Feedback Summary (last 90 days):**
    {"- " + chr(10).join(recent_reviews[:20]) if recent_reviews else "No recent reviews with text."}

    **Instructions:**
    Based *only* on the data provided, generate the following:
    1. **Executive Summary:** A 2-3 sentence overview.
    2. **Dominant Positive Themes (Pros):** A bulleted list of the main positive themes.
    3. **Key Areas for Improvement (Cons):** A bulleted list of the main negative themes.
    4. **Recency Analysis:** A short paragraph describing what recent customers are talking about.
    5. **Strategic Recommendation:** A concluding sentence on what the business should focus on.
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": final_prompt}], temperature=0.4)
        return business_name, response.choices[0].message.content
    except Exception as e:
        return business_name, f"Failed to generate final report: {e}"

# --- STREAMLIT UI AND MAIN WORKFLOW ---

def main():
    st.title("Scalable AI Review Analyzer with Embeddings")

    with st.expander("ℹ️ How This Tool Works", expanded=True):
        st.markdown(...) # Same as before

    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataForSEO API Login", type="password")
        api_password = st.text_input("DataForSEO API Password", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.header("Analysis Settings")
        depth = st.number_input("Reviews to Fetch per Profile", 100, 10000, 500, 100)
        num_clusters = st.slider("Number of Themes to Extract", 4, 10, 7)

    st.markdown("### 1. Enter Google Business Profiles")
    profiles_input = st.text_area("One per line: **Business Name, Location**", height=120)
    
    if st.button("🚀 Run Advanced Analysis", use_container_width=True):
        if not all([api_login, api_password, openai_api_key, profiles_input.strip()]):
            st.error("Please provide all API credentials and at least one GBP profile.")
            return
        
        tasks_list = [
            {"keyword": ", ".join(p[:-1]), "location_name": p[-1]} if len(p) > 1 else {"keyword": p[0], "location_name": "United States"}
            for line in profiles_input.strip().splitlines() if (p := [part.strip() for part in line.split(",")])
        ]
        
        with st.spinner("Fetching reviews using efficient batch polling..."):
            # CORRECT: Directly calling the function, no import needed.
            results = get_review_counts_optimized(api_login, api_password, tasks_list, depth)
        
        if not results:
            st.error("Failed to retrieve any review data from the API.")
            return

        all_reviews_df = get_detailed_reviews_dataframe(results)
        if all_reviews_df.empty:
            st.warning("No reviews with text content were found to analyze.")
            return

        st.markdown("---")
        st.header("🤖 AI Analysis Reports")
        
        business_dfs = {name: group for name, group in all_reviews_df.groupby('Business')}
        
        with st.spinner(f"Analyzing {len(business_dfs)} businesses concurrently..."):
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
