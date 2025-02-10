import time
import streamlit as st
import pandas as pd
from client import RestClient  # Ensure that client.py (DataForSEO client) is in your repository

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
                # Assume the first element contains aggregated data and individual reviews in "items"
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

def main():
    st.title("Google Reviews Checker for GBP Profiles")
    st.markdown("""
        **Overview:**  
        This tool leverages the DataForSEO API to retrieve review counts and detailed review data for your Google Business Profiles (GBP).  
        It shows both the total review count (as listed on Google) and the number of reviews scraped from the page.  
        Additionally, you can download the individual review details as a CSV file for further analysis.  
        
        **How to Use:**  
        1. **Enter Your DataForSEO Credentials** in the sidebar.  
        2. **Enter GBP Profiles:** Provide one GBP profile per line in the format:  
           **Business Name, Location**  
           For example:  
           ```
           Pella Windows and Doors Showroom of Chesterfield, MO, United States
           Pella Windows and Doors Showroom of Bentonville, AR, United States
           ```  
           *Ensure the business name is entered exactly as it appears in your Google Business Profile for accurate results.*  
        3. **Processing Time:**  
           The API tasks are processed asynchronously and may take **2–5 minutes** (or longer if you set a high review depth).  
           The elapsed time is shown during processing.  
        4. **Results:**  
           When complete, a summary table is displayed. You can also download detailed review data (including review text) as a CSV file.
    """)
    
    # Sidebar: API Credentials and Task Settings.
    st.sidebar.header("DataForSEO API Credentials")
    api_login = st.sidebar.text_input("API Login", type="password")
    api_password = st.sidebar.text_input("API Password", type="password")
    
    st.sidebar.header("Task Settings")
    depth = st.sidebar.number_input("Depth (number of reviews to fetch)", min_value=10, max_value=1000, value=100, step=10)
    
    st.markdown("### Enter Your GBP Profiles")
    st.markdown("""
        **Format:** One GBP profile per line in the format:  
        **Business Name, Location**  
        
        **Example:**  
        ```
        Pella Windows and Doors Showroom of Chesterfield, MO, United States
        Pella Windows and Doors Showroom of Bentonville, AR, United States
        ```  
        If you enter only a business name, a default location of 'United States' will be assumed.
    """)
    
    profiles_input = st.text_area("GBP Profiles", height=150)
    
    if st.button("Run Review Check"):
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
                # If there are multiple commas, assume that all parts except the last form the business name.
                keyword = ", ".join(parts[:-1])
                location = parts[-1]
            else:
                keyword = parts[0]
                location = "United States"  # Default location if not specified.
            tasks_list.append({
                "keyword": keyword,
                "location_name": location
            })
        
        with st.spinner("Posting tasks and waiting for results..."):
            results = get_review_counts(api_login, api_password, tasks_list, depth=depth)
        if results is not None:
            # Display summary table.
            summary = parse_results(results)
            st.write("### Summary Results")
            st.table(summary)
            
            # Build detailed reviews DataFrame.
            detailed_df = get_detailed_reviews_dataframe(results)
            if not detailed_df.empty:
                st.write("### Detailed Reviews")
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
            st.error("Failed to retrieve results.")

if __name__ == "__main__":
    main()
