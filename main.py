import time
import streamlit as st
from client import RestClient  # Make sure this file is in your working directory

def get_review_counts(api_login, api_password):
    """
    Post tasks to DataForSEO for the three Pella showrooms,
    then poll for task completion and retrieve the results.
    """
    # Instantiate the REST client with provided credentials
    client = RestClient(api_login, api_password)
    
    # Build POST data – one task per location.
    # You may adjust "depth" (number of reviews to retrieve) as needed.
    post_data = {}
    post_data[len(post_data)] = {
        "keyword": "Pella Windows and Doors Showroom of Chesterfield, MO",
        "location_name": "Chesterfield, MO, United States",
        "language_name": "English",
        "depth": 100  # adjust this value as needed
    }
    post_data[len(post_data)] = {
        "keyword": "Pella Windows and Doors Showroom of Bentonville, AR",
        "location_name": "Bentonville, AR, United States",
        "language_name": "English",
        "depth": 100
    }
    post_data[len(post_data)] = {
        "keyword": "Pella Windows and Doors Showroom of North Little Rock, AR",
        "location_name": "North Little Rock, AR, United States",
        "language_name": "English",
        "depth": 100
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
    # In a production environment, you might want to add more robust timeout/retry logic.
    while True:
        tasks_ready = client.get("/v3/business_data/google/reviews/tasks_ready")
        if (tasks_ready.get("status_code") == 20000 and
            tasks_ready.get("tasks_count", 0) >= len(post_data)):
            break
        time.sleep(10)  # wait 10 seconds between polls
        st.write("Still waiting…")
    
    st.write("Tasks are ready. Retrieving results…")
    results = []
    # Loop through each ready task, then for each task’s result entry, use the provided endpoint to get detailed results.
    for task in tasks_ready.get("tasks", []):
        for resultTaskInfo in task.get("result", []):
            endpoint = resultTaskInfo.get("endpoint")
            if endpoint:
                res = client.get(endpoint)
                results.append(res)
    return results

def parse_results(results):
    """
    Parse each result to extract:
      - The business name (echoed in the task's data)
      - The listing review count (reviews_count)
      - The number of reviews scraped (items_count)
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
    st.title("Google Reviews Checker for Pella Showrooms")
    st.markdown("""
        This self‐service app uses DataForSEO’s API to pull review counts for selected 
        Pella Windows and Doors showrooms. It displays both the total reviews as reported 
        on Google (the listing count) and the number of reviews scraped (the items count). 
        Discrepancies between these numbers may indicate missing reviews.
    """)

    st.sidebar.header("DataForSEO API Credentials")
    api_login = st.sidebar.text_input("API Login", type="password")
    api_password = st.sidebar.text_input("API Password", type="password")

    if st.sidebar.button("Run Review Check"):
        if not api_login or not api_password:
            st.error("Please enter your API credentials in the sidebar.")
        else:
            with st.spinner("Posting tasks and waiting for results…"):
                results = get_review_counts(api_login, api_password)
            if results is not None:
                parsed = parse_results(results)
                st.write("### Results")
                st.table(parsed)
            else:
                st.error("Failed to retrieve results.")

if __name__ == "__main__":
    main()
