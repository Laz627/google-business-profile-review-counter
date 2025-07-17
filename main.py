import time
import streamlit as st
import pandas as pd
from client import RestClient
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# --- NLTK Setup ---
# Download required NLTK data files if not already present.
# This is done once and the results are cached.
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the function to ensure data is downloaded
download_nltk_data()

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
    The DataFrame will include:
      - Business (the 'keyword' from the task data)
      - Location (the 'location_name' from the task data)
      - Timestamp (when the review was posted)
      - Profile Name (who posted the review)
      - Rating (the review's rating value)
      - Review Body (the text content of the review)
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
                    rating = ""
                    if item.get("rating") and isinstance(item["rating"], dict):
                        rating = item["rating"].get("value", "")
                    review_body = item.get("review_text", "")
                    detailed_rows.append({
                        "Business": business,
                        "Location": location,
                        "Timestamp": timestamp,
                        "Profile Name": profile_name,
                        "Rating": rating,
                        "Review Body": review_body
                    })
    if detailed_rows:
        return pd.DataFrame(detailed_rows)
    else:
        return pd.DataFrame()

def analyze_reviews(df):
    """
    Performs sentiment analysis on the 'Review Body' column and extracts
    common themes for pros and cons.
    """
    if df.empty or 'Review Body' not in df.columns:
        return None

    sid = SentimentIntensityAnalyzer()
    
    # Calculate sentiment for each review
    df['sentiment'] = df['Review Body'].apply(lambda review: sid.polarity_scores(str(review))['compound'])
    
    # Classify reviews
    df['sentiment_type'] = df['sentiment'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral'))
    
    # Overall sentiment
    average_sentiment = df['sentiment'].mean()
    if average_sentiment >= 0.05:
        overall_sentiment = "Positive"
    elif average_sentiment <= -0.05:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    # Extract common words from positive and negative reviews
    stop_words = set(stopwords.words('english'))
    
    def get_common_phrases(text_series, top_n=10):
        # Ensure all items are strings before processing
        all_text = ' '.join(str(s) for s in text_series if str(s).strip())
        if not all_text:
            return []
        
        words = word_tokenize(all_text.lower())
        # Filter out stopwords and non-alphabetic characters
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        return [word for word, count in Counter(filtered_words).most_common(top_n)]

    positive_reviews = df[df['sentiment_type'] == 'Positive']['Review Body']
    negative_reviews = df[df['sentiment_type'] == 'Negative']['Review Body']

    pros = get_common_phrases(positive_reviews)
    cons = get_common_phrases(negative_reviews)

    return {
        "overall_sentiment": overall_sentiment,
        "average_score": f"{average_sentiment:.2f}",
        "sentiment_distribution": df['sentiment_type'].value_counts(normalize=True).to_dict(),
        "pros": pros,
        "cons": cons,
    }


def main():
    st.title("Google Reviews Analyzer for GBP Profiles")
    st.markdown("""
        **Overview:**  
        This tool uses the DataForSEO API to fetch, analyze, and summarize reviews for your Google Business Profiles (GBP).
        It provides sentiment analysis, a list of pros and cons, and an overall summary to help you assess customer feedback.
        
        **How to Use:**  
        1. **Enter Your DataForSEO Credentials** in the sidebar.  
        2. **Enter GBP Profiles:** Provide one GBP profile per line in the format:  
           **Business Name, Location**  
           For example:  
           ```
           Pella Windows and Doors Showroom of Chesterfield, MO, United States
           Pella Windows and Doors Showroom of Bentonville, AR, United States
           ```  
        3. **Processing Time:**  
           Tasks may take **2–5 minutes**. The elapsed time is shown during processing.  
        4. **Results:**  
           A summary table, detailed review data, and a full analysis with sentiment, pros, and cons will be displayed.
    """)
    
    # Sidebar: API Credentials and Task Settings.
    st.sidebar.header("DataForSEO API Credentials")
    api_login = st.sidebar.text_input("API Login", type="password")
    api_password = st.sidebar.text_input("API Password", type="password")
    
    st.sidebar.header("Task Settings")
    depth = st.sidebar.number_input("Depth (number of reviews to fetch)", min_value=10, max_value=10000, value=100, step=10)
    
    st.markdown("### Enter Your GBP Profiles")
    st.markdown("""
        **Format:** One GBP profile per line in the format:  
        **Business Name, Location**
    """)
    
    profiles_input = st.text_area("GBP Profiles", height=150, placeholder="Business Name, Location")
    
    if st.button("Run Review Analysis"):
        if not api_login or not api_password:
            st.error("Please enter your API credentials in the sidebar.")
            return
        if not profiles_input.strip():
            st.error("Please enter at least one GBP profile.")
            return
        
        # Parse the input into a list of tasks.
        tasks_list = []
        lines = profiles_input.splitlines()
        for line in lines:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                keyword = ", ".join(parts[:-1])
                location = parts[-1]
            else:
                keyword = parts[0]
                location = "United States"
            tasks_list.append({
                "keyword": keyword,
                "location_name": location
            })
        
        with st.spinner("Posting tasks and waiting for results..."):
            results = get_review_counts(api_login, api_password, tasks_list, depth=depth)
        
        if results:
            st.success("Tasks completed. Generating analysis...")
            
            # Display summary table.
            summary = parse_results(results)
            st.write("### Scrape Summary")
            st.table(summary)
            
            # Build detailed reviews DataFrame.
            detailed_df = get_detailed_reviews_dataframe(results)
            
            if not detailed_df.empty:
                # --- Analysis Section ---
                st.markdown("---")
                st.header("Review Analysis")

                # Analyze each business separately
                for business_name in detailed_df['Business'].unique():
                    st.subheader(f"Analysis for: {business_name}")
                    business_df = detailed_df[detailed_df['Business'] == business_name]
                    
                    analysis = analyze_reviews(business_df)

                    if analysis:
                        # Overall Sentiment
                        st.metric(label="Overall Sentiment", value=analysis['overall_sentiment'])
                        
                        # Pros and Cons
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### Pros 👍")
                            if analysis['pros']:
                                for pro in analysis['pros']:
                                    st.markdown(f"- {pro.capitalize()}")
                            else:
                                st.markdown("No significant positive themes found.")
                        
                        with col2:
                            st.markdown("#### Cons 👎")
                            if analysis['cons']:
                                for con in analysis['cons']:
                                    st.markdown(f"- {con.capitalize()}")
                            else:
                                st.markdown("No significant negative themes found.")
                        
                        # Final Summary
                        st.markdown("#### Summary & Recommendation")
                        num_reviews = len(business_df)
                        avg_rating = pd.to_numeric(business_df['Rating'], errors='coerce').mean()
                        
                        summary_text = f"""
                        Based on the analysis of **{num_reviews}** reviews with an average rating of **{avg_rating:.2f} stars**, 
                        the sentiment towards **{business_name}** is generally **{analysis['overall_sentiment']}**.

                        Key positive themes include **{', '.join(analysis['pros'][:3])}**. 
                        However, some customers have raised concerns about **{', '.join(analysis['cons'][:3])}**.

                        **Recommendation:** Potential customers should feel confident, particularly if the positive aspects align with their priorities. 
                        It may be wise to inquire about the common issues mentioned to ensure a smooth experience.
                        """
                        st.info(summary_text)

                    else:
                        st.warning(f"Could not generate an analysis for {business_name}. Not enough review text available.")
                
                # --- Detailed Data Section ---
                st.markdown("---")
                st.header("Detailed Review Data")
                st.dataframe(detailed_df)
                
                # Provide a CSV download button.
                csv = detailed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Detailed Reviews as CSV",
                    data=csv,
                    file_name='detailed_reviews.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No detailed review data was found.")
        else:
            st.error("Failed to retrieve results from the API.")

if __name__ == "__main__":
    main()
