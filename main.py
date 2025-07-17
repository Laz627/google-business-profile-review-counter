import time
import streamlit as st
import pandas as pd
from client import RestClient
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

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

        # Check the status of all tasks in a single API call
        ready_response = client.get("/v3/business_data/google/reviews/tasks_ready")
        if ready_response.get("status_code") != 20000:
            st.warning("Could not check task readiness. Will retry...")
            time.sleep(20)
            continue

        ready_tasks = ready_response.get("tasks", [])[0].get("result", [])
        if not ready_tasks:
            time.sleep(20) # Wait before checking again
            continue
        
        for task_info in ready_tasks:
            task_id = task_info.get("id")
            if task_id and task_id in task_ids and task_id not in completed_task_ids:
                st.write(f"Task {task_id} is ready. Fetching results...")
                # Fetch the result for the completed task
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
    For each completed task result, extract the individual review items and build a DataFrame.
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
                review_info = results_array[0]
                items = review_info.get("items", [])
                for item in items:
                    rating_value = item.get("rating", {}).get("value", "") if isinstance(item.get("rating"), dict) else ""
                    review_body = item.get("review_text", "")
                    
                    if review_body: # Only include reviews with text
                        detailed_rows.append({
                            "Business": business, "Location": location,
                            "Rating": rating_value, "Review Body": review_body
                        })
    return pd.DataFrame(detailed_rows) if detailed_rows else pd.DataFrame()


def analyze_reviews_with_gpt(api_key, business_name, reviews_df, model="gpt-4o-mini"):
    """
    Analyzes reviews for a SINGLE business using an OpenAI model.
    """
    if reviews_df.empty:
        return business_name, "No reviews with text content were found to analyze."

    reviews_text = "\n---\n".join(
        f"Rating: {row['Rating']}/5\nReview: {row['Review Body']}"
        for _, row in reviews_df.iterrows()
    )

    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return business_name, f"Error initializing OpenAI client: {e}"

    prompt = f"""
    You are a business analyst. Analyze the following customer reviews for '{business_name}'.
    Based *only* on the reviews, provide the following in clear markdown:

    1. **Overall Sentiment**: A single word (Positive, Negative, Mixed).
    2. **Executive Summary**: A 2-3 sentence paragraph summarizing key feedback.
    3. **Strong Pros**: A bulleted list of the 3-5 most common positive themes.
    4. **Key Cons**: A bulleted list of the 3-5 most common negative themes.
    5. **Final Recommendation**: A concluding sentence for potential customers.

    Reviews:
    ---
    {reviews_text}
    ---
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful business review analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return business_name, response.choices[0].message.content
    except Exception as e:
        return business_name, f"An error occurred with the OpenAI API: {e}"

def main():
    st.set_page_config(layout="wide")
    st.title("🚀 High-Speed Generative AI Review Analyzer")
    st.markdown("This tool uses **batch polling** and **concurrent AI analysis** for maximum speed.")

    # --- Sidebar Setup ---
    with st.sidebar:
        st.header("API Credentials")
        api_login = st.text_input("DataForSEO API Login", type="password")
        api_password = st.text_input("DataForSEO API Password", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        
        st.header("Task Settings")
        depth = st.number_input("Depth (reviews to fetch)", 10, 1000, 100, 10)
    
    st.markdown("### 1. Enter GBP Profiles")
    profiles_input = st.text_area("One per line: **Business Name, Location**", height=120, placeholder="Pella Windows and Doors, Chesterfield, MO, United States")
    
    if st.button("⚡ Run High-Speed Analysis", use_container_width=True):
        if not all([api_login, api_password, openai_api_key, profiles_input.strip()]):
            st.error("Please provide all API credentials and at least one GBP profile.")
            return
        
        tasks_list = [
            {"keyword": ", ".join(parts[:-1]), "location_name": parts[-1]} if len(parts) > 1 else {"keyword": parts[0], "location_name": "United States"}
            for line in profiles_input.strip().splitlines() if (parts := [p.strip() for p in line.split(",")])
        ]
        
        # --- Optimized Data Fetching ---
        with st.spinner("Fetching reviews using efficient batch polling..."):
            results = get_review_counts_optimized(api_login, api_password, tasks_list, depth)
        
        if not results:
            st.error("Failed to retrieve review data.")
            return

        all_reviews_df = get_detailed_reviews_dataframe(results)
        if all_reviews_df.empty:
            st.warning("No reviews with text were found to analyze.")
            return

        # --- Concurrent AI Analysis ---
        st.markdown("---")
        st.header("🤖 AI-Powered Review Analysis")
        
        # Group reviews by business
        business_dfs = {name: group for name, group in all_reviews_df.groupby('Business')}
        
        with st.spinner(f"Analyzing {len(business_dfs)} businesses concurrently..."):
            # Use a ThreadPoolExecutor to run analyses in parallel
            with ThreadPoolExecutor(max_workers=10) as executor:
                # Create a future for each business analysis
                future_to_business = {
                    executor.submit(analyze_reviews_with_gpt, openai_api_key, name, df): name
                    for name, df in business_dfs.items()
                }

                # Create placeholders for results
                placeholders = {name: st.empty() for name in business_dfs.keys()}
                
                # Process results as they are completed
                for future in as_completed(future_to_business):
                    business_name, analysis_result = future.result()
                    with placeholders[business_name].container():
                        st.subheader(f"Analysis for: {business_name}")
                        st.markdown(analysis_result)

        # --- Data Download ---
        st.markdown("---")
        st.header("📄 Detailed Review Data")
        st.dataframe(all_reviews_df)
        csv = all_reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Reviews as CSV", csv, "detailed_reviews.csv", "text/csv")

if __name__ == "__main__":
    main()
