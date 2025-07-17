import time
import streamlit as st
import pandas as pd
from client import RestClient
import openai  # Import the OpenAI library

# --- OpenAI Client Setup ---
# The API key is handled via the Streamlit sidebar input
# No global client is instantiated here to allow key entry in the UI

def get_review_counts(api_login, api_password, tasks_list, depth=100):
    """
    Build tasks from a list of dictionaries (each with 'keyword' and 'location_name'),
    post them to DataForSEO, and then poll each task individually (using its ID)
    until the result is available.
    """
    client = RestClient(api_login, api_password)
    
    # Build the POST data from the tasks_list.
    post_data = {}
    for idx, task in enumerate(tasks_list):
        post_data[idx] = {
            "keyword": task["keyword"],
            "location_name": task["location_name"],
            "language_name": "English",
            "depth": depth
        }
        
    st.write("**Posting tasks...**")
    response = client.post("/v3/business_data/google/reviews/task_post", post_data)
    if response.get("status_code") != 20000:
        st.error("Error posting tasks. Code: {} Message: {}".format(
            response.get("status_code"), response.get("status_message")))
        return None
    
    # Extract task IDs from the response.
    task_ids = []
    for task in response.get("tasks", []):
        task_id = task.get("id")
        if task_id:
            task_ids.append(task_id)
    st.write("Task IDs:", task_ids)
    
    # Poll each task individually using its task_get endpoint.
    st.write("**Polling tasks for completion...**")
    completed_results = {}
    start_time = time.time()
    polling_status = st.empty()  # Placeholder to update elapsed time
    while len(completed_results) < len(task_ids):
        for task_id in task_ids:
            if task_id in completed_results:
                continue  # Already completed
            result = client.get("/v3/business_data/google/reviews/task_get/" + task_id)
            if result.get("status_code") == 20000:
                tasks_list_result = result.get("tasks", [])
                if tasks_list_result:
                    task_data = tasks_list_result[0]
                    # Check if "result" exists and is non-empty.
                    if task_data.get("result"):
                        completed_results[task_id] = result
                        st.write(f"Task {task_id} completed.")
        elapsed_time = int(time.time() - start_time)
        polling_status.text(f"**Time elapsed:** {elapsed_time} seconds")
        if len(completed_results) < len(task_ids):
            time.sleep(10)
            
    # Return the list of completed results.
    return list(completed_results.values())

def parse_results(results):
    """
    Create a summary list for each result that shows:
      - Business (from the task 'keyword')
      - Total Reviews (listing count from Google)
      - Scraped Reviews (items count from the API response)
      - Whether there is a discrepancy.
    """
    parsed = []
    for res in results:
        if res.get("status_code") == 20000:
            task_result = res.get("tasks", [])[0]
            task_params = task_result.get("data", {})
            results_array = task_result.get("result", [])
            if results_array:
                review_info = results_array[0]
                keyword = task_params.get("keyword", "Unknown")
                total_reviews = review_info.get("reviews_count")
                items_count = review_info.get("items_count")
                parsed.append({
                    "Business": keyword,
                    "Total Reviews (listing count)": total_reviews,
                    "Scraped Reviews (items count)": items_count,
                    "Discrepancy": "Yes" if total_reviews != items_count else "No"
                })
    return parsed

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
                    timestamp = item.get("timestamp", "")
                    profile_name = item.get("profile_name", "")
                    rating_value = item.get("rating", {}).get("value", "") if isinstance(item.get("rating"), dict) else ""
                    review_body = item.get("review_text", "")
                    
                    # Only include reviews with text for analysis
                    if review_body:
                        detailed_rows.append({
                            "Business": business,
                            "Location": location,
                            "Timestamp": timestamp,
                            "Profile Name": profile_name,
                            "Rating": rating_value,
                            "Review Body": review_body
                        })
    if detailed_rows:
        return pd.DataFrame(detailed_rows)
    return pd.DataFrame()

def analyze_reviews_with_gpt(api_key, reviews_df, model="gpt-4o-mini"):
    """
    Analyzes reviews using an OpenAI model to generate a summary, pros, cons, and sentiment.
    """
    if reviews_df.empty:
        return "No reviews with text content were found to analyze."

    # Concatenate reviews into a single block of text
    # We include ratings to give the model more context
    reviews_text = "\n---\n".join(
        f"Rating: {row['Rating']}/5\nReview: {row['Review Body']}"
        for _, row in reviews_df.iterrows()
    )

    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        return f"Error initializing OpenAI client: {e}"

    # Construct the prompt for the generative model
    prompt = f"""
    You are a business analyst specialized in customer feedback. Analyze the following customer reviews for a business.
    Based *only* on the information in these reviews, provide the following in clear, well-structured markdown format:

    1.  **Overall Sentiment**: A single word (e.g., Positive, Negative, Mixed).
    2.  **Executive Summary**: A concise, 2-3 sentence paragraph summarizing the key takeaways from the customer feedback.
    3.  **Strong Pros**: A bulleted list of the 3-5 most common positive themes or compliments.
    4.  **Key Cons**: A bulleted list of the 3-5 most common negative themes or complaints.
    5.  **Final Recommendation**: A concluding sentence to help a potential customer decide if they should do business here based on the reviews.

    Here are the reviews:
    ---
    {reviews_text}
    ---
    """
    
    try:
        st.info(f"Sending {len(reviews_df)} reviews to {model} for analysis...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful business review analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Lower temperature for more factual, less creative output
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred while communicating with the OpenAI API: {e}"

def main():
    st.title("Generative AI Google Reviews Analyzer")
    st.markdown("""
        **Overview:**  
        This tool uses DataForSEO to fetch Google reviews and **OpenAI's GPT model** to perform an in-depth analysis.
        It delivers a human-like summary, identifies key pros and cons, and assesses overall sentiment to provide actionable business insights.
        
        **How to Use:**  
        1.  **Enter API Credentials** in the sidebar (DataForSEO and OpenAI).
        2.  **Enter GBP Profiles** one per line: **Business Name, Location**.
        3.  **Run Analysis:** Processing takes **2–5 minutes**.
        4.  **Get Results:** Review the AI-generated analysis for each business.
    """)
    
    # --- Sidebar Setup ---
    st.sidebar.header("API Credentials")
    api_login = st.sidebar.text_input("DataForSEO API Login", type="password")
    api_password = st.sidebar.text_input("DataForSEO API Password", type="password")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    st.sidebar.header("Task Settings")
    depth = st.sidebar.number_input("Depth (reviews to fetch)", min_value=10, max_value=1000, value=100, step=10)
    
    st.markdown("### Enter Your GBP Profiles")
    profiles_input = st.text_area(
        "GBP Profiles (one per line)", 
        height=120, 
        placeholder="Pella Windows and Doors, Chesterfield, MO, United States"
    )
    
    if st.button("Run Generative AI Analysis"):
        if not all([api_login, api_password, openai_api_key]):
            st.error("Please enter all API credentials in the sidebar.")
            return
        if not profiles_input.strip():
            st.error("Please enter at least one GBP profile.")
            return
        
        # Parse input into tasks
        tasks_list = []
        for line in profiles_input.strip().splitlines():
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            keyword, location = (", ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else (parts[0], "United States")
            tasks_list.append({"keyword": keyword, "location_name": location})
        
        # --- Data Fetching ---
        with st.spinner("Fetching reviews from DataForSEO..."):
            results = get_review_counts(api_login, api_password, tasks_list, depth=depth)
        
        if not results:
            st.error("Failed to retrieve review data. Please check credentials or task settings.")
            return

        st.success("Review data fetched. Preparing for analysis.")
        detailed_df = get_detailed_reviews_dataframe(results)

        if detailed_df.empty:
            st.warning("No reviews with text content were found to analyze.")
            return

        # --- Analysis and Display ---
        st.markdown("---")
        st.header("AI-Powered Review Analysis")

        for business_name in detailed_df['Business'].unique():
            st.subheader(f"Analysis for: {business_name}")
            business_df = detailed_df[detailed_df['Business'] == business_name].copy()
            
            with st.spinner(f"Analyzing reviews for {business_name} with GPT..."):
                analysis_result = analyze_reviews_with_gpt(openai_api_key, business_df)
            
            st.markdown(analysis_result)

        # --- Data Download Section ---
        st.markdown("---")
        if not detailed_df.empty:
            st.header("Detailed Review Data")
            st.dataframe(detailed_df)
            csv = detailed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Detailed Reviews as CSV",
                csv,
                "detailed_reviews.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()
