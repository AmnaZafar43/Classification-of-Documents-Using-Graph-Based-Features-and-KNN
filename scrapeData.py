from bs4 import BeautifulSoup
import requests

# URL of the page to scrape
url = "https://oercommons.org/courses/22-years-of-sea-surface-temperatures-2?__hub_id=100"

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find the material-details element
material_details = soup.find('div', class_='material-details')

if material_details:
    # Extract description
    description = material_details.find('dd', itemprop='description').text.strip()

    # Extract subjects
    subjects = material_details.find_all('span', itemprop='about')
    subjects_list = [subject.text.strip() for subject in subjects]

    print("Description:", description)
    print("Subjects:", subjects_list)
else:
    print("Material details not found.")
