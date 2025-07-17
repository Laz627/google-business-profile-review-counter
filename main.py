import time
import streamlit as st
import pandas as pd
import numpy as np
import json
from client import RestClient # Ensure client.py is in the repository
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="Interactive AI Review Analyzer")

# --- DATA FETCHING & CORE ANALYSIS (UNCHANGED) ---
# All functions from the previous script are included here without modification.
# For brevity, their code is collapsed in this view but is present in the full script.

def get_review_counts_optimized(api_login, api_password, tasks_list, depth=100):
    client = RestClient(api_login, api_password)
    post_data = {idx: {"keyword": task["keyword"], "location_name": task["location_name"], "language_name": "English", "depth": depth} for idx, task in enumerate(tasks_list)}
    with st.status("Fetching reviews from Google...", expanded=True) as status:
        status.write("Posting tasks to DataForSEO API...")
        post_response = client.post("/v3/business_data/google/reviews/task_post", post_data)
        if post_response.get("status_code") != 20000:
            status.update(label="API Error!", state="error", expanded=True); st.error(f"Error: {post_response.get('status_message')}"); return None
        task_ids = [task["id"] for task in post_response.get("tasks", []) if task.get("id")]
        if not task_ids:
            status.update(label="Task ID Error!", state="error", expanded=True); st.error("Could not retrieve task IDs."); return None
        status.write(f"Tasks posted. IDs: {task_ids}")
        status.write("Polling for results...")
        completed_results, completed_task_ids, start_time = {}, set(), time.time()
        while len(completed_task_ids) < len(task_ids):
            elapsed_time = int(time.time() - start_time)
            status.write(f"**Time elapsed:** {elapsed_time}s. Completed {len(completed_task_ids)}/{len(task_ids)}.")
            ready_response = client.get("/v3/business_data/google/reviews/tasks_ready")
            if ready_response.get("status_code") != 20000: time.sleep(20); continue
            for task_info in ready_response.get("tasks", [])[0].get("result", []):
                task_id = task_info.get("id")
                if task_id in task_ids and task_id not in completed_task_ids:
                    status.write(f"Task {task_id} is ready...")
                    result_response = client.get(f"/v3/business_data/google/reviews/task_get/{task_id}")
                    if result_response.get("status_code") == 20000: completed_results[task_id] = result_response; completed_task_ids.add(task_id)
            if len(completed_task_ids) < len(task_ids): time.sleep(20)
        status.update(label="All tasks completed!", state="complete", expanded=False)
    return list(completed_results.values())

def get_detailed_reviews_dataframe(results):
    detailed_rows = []
    for res in results:
        if res.get("status_code") == 20000:
            task_result = res.get("tasks", [])[0]
            business_name = task_result.get("data", {}).get("keyword", "Unknown")
            location_name = task_result.get("data", {}).get("location_name", "Unknown")
            items = task_result.get("result", [{}])[0].get("items", [])
            for item in items:
                if review_body := item.get("review_text", ""):
                    detailed_rows.append({"Business": business_name, "Location": location_name, "Rating": item.get("rating", {}).get("value"), "Review Body": review_body, "Timestamp": item.get("timestamp")})
    return pd.DataFrame(detailed_rows)

def get_embeddings(api_key, texts, model="text-embedding-3-small"):
    try:
        client = openai.OpenAI(api_key=api_key); response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Failed to generate embeddings: {e}"); return None

def summarize_cluster_theme(api_key, reviews_sample, model="gpt-4o-mini"):
    prompt = f"Analyze these reviews. What is the dominant theme? Respond with a short theme name (4-6 words). Example: 'Fast and friendly customer service'.\n\nReviews:\n" + "\n".join("- " + r for r in reviews_sample)
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=[{"role": "system", "content": "You are an expert at identifying themes."}, {"role": "user", "content": prompt}], temperature=0.2)
        return response.choices[0].message.content.strip()
    except Exception: return "Unnamed Theme"

def analyze_reviews_with_embeddings(api_key, business_name, reviews_df, num_clusters=7, model="gpt-4o-mini"):
    if reviews_df.empty or len(reviews_df) < num_clusters:
        return business_name, {"error": "Not enough reviews for a detailed analysis."}, None
    
    review_texts = reviews_df['Review Body'].tolist()
    embeddings = get_embeddings(api_key, review_texts)
    if not embeddings: return business_name, {"error": "Could not generate embeddings."}, None
    reviews_df['embedding'] = embeddings

    actual_clusters = min(num_clusters, len(review_texts))
    kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto')
    reviews_df['cluster'] = kmeans.fit_predict(np.array(embeddings))
    reviews_df['rating_numeric'] = pd.to_numeric(reviews_df['Rating'], errors='coerce')
    
    cluster_themes = {}
    with ThreadPoolExecutor(max_workers=actual_clusters) as executor:
        future_to_cluster = {executor.submit(summarize_cluster_theme, api_key, reviews_df[reviews_df['cluster'] == i]['Review Body'].head(10).tolist()): i for i in range(actual_clusters)}
        for future in as_completed(future_to_cluster): cluster_themes[future_to_cluster[future]] = future.result()
    
    positive_themes, negative_themes = [], []
    for i, theme_name in cluster_themes.items():
        cluster_data = reviews_df[reviews_df['cluster'] == i]; avg_rating = cluster_data['rating_numeric'].mean()
        theme_obj = {"name": theme_name, "rating": avg_rating, "prevalence": len(cluster_data) / len(reviews_df)}
        if avg_rating >= 3.5: positive_themes.append(theme_obj)
        elif avg_rating < 2.5: negative_themes.append(theme_obj)
        
    reviews_df['Timestamp'] = pd.to_datetime(reviews_df['Timestamp'], errors='coerce')
    ninety_days_ago = pd.Timestamp.now(tz=reviews_df['Timestamp'].dt.tz) - pd.Timedelta(days=90) if reviews_df['Timestamp'].dt.tz else pd.Timestamp.now() - pd.Timedelta(days=90)
    recent_reviews_text = "\n".join("- " + r for r in reviews_df[reviews_df['Timestamp'] >= ninety_days_ago]['Review Body'].tolist()[:20])
    final_prompt = f"""You are a business analyst. Based on the provided data, write the prose sections for a report on '{business_name}'. Recent reviews are provided for context on current customer sentiment. **Recent Feedback (last 90 days):**\n{recent_reviews_text if recent_reviews_text else "No recent reviews with text."}\n\n**Task:** Generate a JSON object with three keys: "summary", "recency_analysis", "recommendation". - "summary": A 2-3 sentence executive summary of the overall customer feedback. - "recency_analysis": A short paragraph analyzing the themes from the recent feedback provided. - "recommendation": A single, actionable strategic recommendation for the business."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(model=model, response_format={"type": "json_object"}, messages=[{"role": "user", "content": final_prompt}], temperature=0.4)
        report_prose = json.loads(response.choices[0].message.content)
        return business_name, {"prose": report_prose, "pros": positive_themes, "cons": negative_themes}, reviews_df
    except Exception as e:
        return business_name, {"error": f"Failed to generate final report: {e}"}, None

def answer_question_with_rag(api_key, user_question, business_reviews_df, model="gpt-4o-mini"):
    st.session_state['qna_answer'] = ""
    with st.spinner("Finding relevant reviews and generating an answer..."):
        question_embedding = get_embeddings(api_key, [user_question])[0]
        review_embeddings = np.array(business_reviews_df['embedding'].tolist())
        similarities = [1 - cosine(question_embedding, emb) for emb in review_embeddings]
        top_indices = np.argsort(similarities)[-7:][::-1]
        relevant_reviews = business_reviews_df.iloc[top_indices]['Review Body'].tolist()
        prompt = f"""You are a helpful Q&A assistant. Based ONLY on the provided reviews below, answer the user's question. If the reviews do not contain enough information to answer, you MUST explicitly state that the information is not available. **User's Question:** "{user_question}"\n\n**Provided Reviews:**\n- {"- ".join(relevant_reviews)}\n\n**Answer:**"""
        try:
            client = openai.OpenAI(api_key=api_key); response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0)
            st.session_state['qna_answer'] = response.choices[0].message.content
        except Exception as e:
            st.session_state['qna_answer'] = f"An error occurred: {e}"

# --- STREAMLIT UI AND MAIN WORKFLOW ---
def main():
    st.title("Interactive AI Review Analyzer")

    if 'analysis_complete' not in st.session_state:
        st.session_state.update({'analysis_complete': False, 'business_data': {}, 'qna_answer': "", 'all_reviews_df': pd.DataFrame()})

    with st.expander("🔎 How to Use This Tool", expanded=True):
        # ... user guide text unchanged ...
        st.markdown("""
        This tool helps you go beyond simple star ratings to understand the ***why*** behind customer feedback. Using advanced AI, it analyzes hundreds of reviews to automatically identify key themes, assess sentiment, and provide actionable insights.
        #### What You'll Need
        1.  **DataForSEO:** To retrieve review data from Google. You'll need an **API Login** and **API Password**.
        2.  **OpenAI:** To provide the AI models for analysis. You'll need an **OpenAI API Key**.
        ---
        #### Step-by-Step Guide
        **1. Configure Your Analysis (in the sidebar)**
        **2. Enter Business Profiles** (`Business Name, Location`)
        **3. Run the Analysis & Interpret the Report**
        **4. Ask Follow-up Questions** using the interactive Q&A section.
        """)


    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataForSEO API Login", type="password")
        api_password = st.text_input("DataForSEO API Password", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.header("Analysis Settings")
        depth = st.number_input("Reviews to Fetch", 100, 10000, 500, 100, help="How many reviews to analyze per business?")
        num_clusters = st.slider("Themes to Extract", 4, 10, 7, help="How many distinct topics should the AI try to find?")

    st.markdown("##### Enter Google Business Profiles")
    profiles_input = st.text_area("One per line: **Business Name, Location**", height=100, placeholder="Pella Windows, Chesterfield, MO\nExample Restaurant, New York, NY")
    
    if st.button("🚀 Run Advanced Analysis", use_container_width=True):
        st.session_state.update({'analysis_complete': False, 'business_data': {}, 'all_reviews_df': pd.DataFrame()})
        if not all([api_login, api_password, openai_api_key, profiles_input.strip()]):
            st.error("Please provide all API credentials and at least one GBP profile."); return
        
        tasks_list = [{"keyword": ", ".join(p[:-1]), "location_name": p[-1]} if len(p) > 1 else {"keyword": p[0], "location_name": "United States"} for line in profiles_input.strip().splitlines() if (p := [part.strip() for part in line.split(",")])]
        results = get_review_counts_optimized(api_login, api_password, tasks_list, depth)
        if not results: st.error("Failed to retrieve data from DataForSEO."); return
        
        all_reviews_df = get_detailed_reviews_dataframe(results)
        if all_reviews_df.empty: st.warning("No reviews with text content were found."); return
        st.session_state['all_reviews_df'] = all_reviews_df

        st.markdown("---"); st.header("🤖 AI Analysis Reports")
        with st.spinner(f"Analyzing reviews for {len(tasks_list)} businesses..."):
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_business = {executor.submit(analyze_reviews_with_embeddings, openai_api_key, name, df.copy(), num_clusters): name for name, df in all_reviews_df.groupby('Business')}
                for future in as_completed(future_to_business):
                    business_name, report_data, reviews_with_embeddings_df = future.result()
                    if reviews_with_embeddings_df is not None:
                        st.session_state['business_data'][business_name] = {"report": report_data, "reviews_df": reviews_with_embeddings_df}
        st.session_state['analysis_complete'] = True

    if st.session_state.get('analysis_complete'):
        # --- Display Reports ---
        for business_name, data in st.session_state['business_data'].items():
            report_data = data['report']
            with st.container(border=True):
                st.subheader(f"Analysis for: {business_name}")
                if "error" in report_data: st.error(report_data["error"]); continue
                prose = report_data.get("prose", {}); st.markdown("##### Executive Summary"); st.write(prose.get("summary", "N/A")); st.divider()
                st.markdown("##### Dominant Positive Themes (Pros)");
                for theme in sorted(report_data.get("pros", []), key=lambda x: x['prevalence'], reverse=True):
                    col1, col2, col3 = st.columns([4, 1, 1]); col1.write(f"**{theme['name']}**"); col2.metric("Avg Rating", f"{theme['rating']:.2f} ⭐"); col3.metric("Prevalence", f"{theme['prevalence']:.1%}")
                st.markdown("##### Key Areas for Improvement (Cons)")
                cons = sorted(report_data.get("cons", []), key=lambda x: x['prevalence'], reverse=True)
                if not cons: st.info("No significant negative themes were identified.")
                else:
                    for theme in cons:
                        col1, col2, col3 = st.columns([4, 1, 1]); col1.write(f"**{theme['name']}**"); col2.metric("Avg Rating", f"{theme['rating']:.2f} ⭐"); col3.metric("Prevalence", f"{theme['prevalence']:.1%}")
                st.divider()
                st.markdown("##### Recency Analysis"); st.write(prose.get("recency_analysis", "N/A"))
                st.markdown("##### Strategic Recommendation"); st.success(f"**Recommendation:** {prose.get('recommendation', 'N/A')}")

        # --- Interactive Q&A Section ---
        st.markdown("---"); st.header("💬 Ask a Follow-up Question")
        business_options = list(st.session_state['business_data'].keys())
        if business_options:
            selected_business = st.selectbox("Select a business to ask about:", business_options)
            user_question = st.text_input("Enter your question:", placeholder="e.g., How was the customer service?")
            if st.button("Get Answer", key="qna_button"):
                if user_question and selected_business:
                    answer_question_with_rag(openai_api_key, user_question, st.session_state['business_data'][selected_business]['reviews_df'])
                else:
                    st.error("Please select a business and enter a question.")
            if st.session_state.get('qna_answer'): st.info(st.session_state['qna_answer'])
        
        # --- NEW: DETAILED DATA EXPORT SECTION ---
        st.markdown("---")
        with st.expander("📄 View and Export All Fetched Review Data"):
            display_df = st.session_state.get('all_reviews_df', pd.DataFrame())
            if not display_df.empty:
                st.dataframe(display_df)
                csv = display_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Reviews as CSV",
                    data=csv,
                    file_name='all_customer_reviews.csv',
                    mime='text/csv',
                )
            else:
                st.write("No data available to display.")


if __name__ == "__main__":
    main()
