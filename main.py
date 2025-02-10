import time
import streamlit as st
from client import RestClient  # Ensure that client.py is in your repository

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
        if len(completed_results) < len(task_ids):
            st.write("Waiting for remaining tasks...")
            time.sleep(10)
            
    # Return the list of completed results.
    return list(completed_results.values())

def parse_results(results):
    """
    Extract the business name (from task data), the total review count as reported
    on Google (reviews_count), and the number of reviews scraped (items_count).
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

def main():
    st.title("Google Reviews Checker for GBP Profiles")
    st.markdown("""
        This self-service app uses DataForSEO’s API to retrieve review counts for Google Business Profiles.
        Enter your GBP details below to see both the total (listing) review count and the number of reviews scraped.
        Discrepancies may indicate missing reviews.
    """)
    
    # API Credentials in the sidebar.
    st.sidebar.header("DataForSEO API Credentials")
    api_login = st.sidebar.text_input("API Login", type="password")
    api_password = st.sidebar.text_input("API Password", type="password")
    
    st.sidebar.header("Task Settings")
    depth = st.sidebar.number_input("Depth (number of reviews to fetch)", min_value=10, max_value=1000, value=100, step=10)
    
    st.markdown("### Enter Your GBP Profiles")
    st.markdown("""
        Enter one GBP profile per line in the format:
        **Business Name, Location**
        
        For example:
        ```
        Pella Windows and Doors Showroom of Chesterfield, MO, United States
        Pella Windows and Doors Showroom of Bentonville, AR, United States
        ```
        If you enter only a business name, a default location of 'United States' will be used.
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
                # If multiple commas are used, assume that all parts except the last form the business name.
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
            parsed = parse_results(results)
            st.write("### Results")
            st.table(parsed)
        else:
            st.error("Failed to retrieve results.")

if __name__ == "__main__":
    main()
