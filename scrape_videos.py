from youtube_search import YoutubeSearch
import csv 
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode


YOUTUBE_URL_PREFIX = 'https://www.youtube.com'

def clean_url(url: str):
    '''
    Standardizes scraped URLs by removing extraneous parameters.

    Allows us to remove duplicate videos effectively.

    Input:
        url (str): a youtube URL
    '''
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Remove the `pp=yg...` or any other unnecessary parameters
    # We'll keep the `v` parameter (the video ID) and discard the rest
    clean_query = parse_qs(parsed_url.query)
    clean_query = {key: value for key, value in clean_query.items() if key == 'v'}
    
    # Rebuild the URL with the clean query
    cleaned_url = urlunparse(parsed_url._replace(query=urlencode(clean_query, doseq=True)))
    
    return cleaned_url

def search_youtube(query: str, max_results = 20):
    '''
    Main function to scrape from YouTube, using the youtube_search package

    Input:
        query (str): the search query to scrape from
        max_results (int): # of results to return, is limited to 20 in this package
    '''
    ## returns a list of dictionaries
    videos_search = YoutubeSearch(query, max_results=max_results).to_dict()
    # Filter results that contain "application" in the title
    filtered_results = [
        (vid["title"], clean_url(YOUTUBE_URL_PREFIX + vid["url_suffix"]))
        for vid in videos_search
        if "application" in vid["title"].lower()
    ]
    return filtered_results

def save_to_csv(data, filename="./data/youtube_search_results.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Optional: Write a header row
        writer.writerow(["title", "link"])
        
        # Write data (convert set to list for consistent ordering if needed)
        writer.writerows(data)

if __name__ == '__main__':
    seasons = {"W": "Winter", "S": "Summer"}
    short_years = range(10, 26)  # For the short year format (10 to 25)
    full_years = range(2010, 2026)  # For the full year format (2010 to 2025)

    # Combining both formats
    list_of_batches = [
        f"{season}{short_year}" for short_year in short_years for season in seasons
    ] + [
        f"{season_name} {full_year}" for full_year in full_years for season_name in seasons.values()
    ]
    list_of_batches.extend(["F24", "Fall 2024"]) ## YC introduced a Fall batch in 2024
    retrieved_videos = set() ## use a set to remove duplicates
    for batch in list_of_batches:
        search_results = search_youtube(f"YC {batch} Application Video")
        retrieved_videos.update(search_results)
    save_to_csv(retrieved_videos)
    print("Results saved to youtube_search_results.csv")