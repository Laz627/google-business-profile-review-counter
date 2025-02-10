import time
import streamlit as st
from client import RestClient  # Ensure client.py is in your repository

def get_review_counts(api_login, api_password, tasks_list, depth=100):
    """
    Build tasks from a list of dictionaries (each with 'keyword' and 'location_name'),
    post them to DataForSEO, poll for completion, and retrieve the results.
    """
    # Instantiate the REST client using user-supplied credentials
    client = RestClient(api_login, api_password)
    
    # Build the POST data dictionary from the user-provided tasks_list.
    post_data = {}
    for idx, task in enumerate(tasks_list):
        post_data[idx] = {
            "keyword": task["keyword"],
            "location_name": task["location_name"],
            "language_name": "English",
            "depth": depth  # Number of reviews to fetch; adjust as needed.
        }
    
    st.write("**Setting tasks…**")
    response = client.post("/v3/business_data/google/reviews/task_post", post_data)
    
    if response.get("status_code") != 20000:
        st.error("Error posting tasks. Code: {} Message: {}".format(
            response.get("status_code"), response.get("status_message")))
        return None
    
    st.write("Tasks have been posted successfully.")
    
    # Poll for completed tasks.
    st.write("**Waiting for tasks to complete…**")
    # (In production, you might add more robust error handling or timeout logic.)
    while True:
        tasks_ready = client.get("/v3/business_data/google/reviews/tasks_ready")
        if (tasks_ready.get("status_code") == 20000 and
            tasks_ready.get("tasks_count", 0) >= len(post_data)):
            break
        time.sleep(10)  # Wait 10 seconds between polls.
        st.write("Still waiting…")
    
    st.write("Tasks are ready. Retrieving results…")
    results = []
    # Retrieve detailed results for each completed task.
    for task in tasks_ready.get("tasks", []):
        for resultTaskInfo in task.get("result", []):
            endpoint = resultTaskInfo.get("endpoint")
            if endpoint:
                res = client.get(endpoint)
                results.append(res)
    return results

def parse_results(results):
    """
    Extract from each result:
      - The business (GBP profile) name (from task data)
      - The total reviews count as reported on the listing (reviews_count)
      - The scraped reviews count (items_count)
      - An indication if there’s a discrepancy.
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
            # Split the line by comma.
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                # If more than one comma is found, assume that all parts except the last form the business name.
                keyword = ", ".join(parts[:-1])
                location = parts[-1]
            else:
                keyword = parts[0]
                location = "United States"  # Default location if not specified.
            tasks_list.append({
                "keyword": keyword,
                "location_name": location
            })
        
        with st.spinner("Posting tasks and waiting for results…"):
            results = get_review_counts(api_login, api_password, tasks_list, depth=depth)
        if results is not None:
            parsed = parse_results(results)
            st.write("### Results")
            st.table(parsed)
        else:
            st.error("Failed to retrieve results.")

if __name__ == "__main__":
    main()
