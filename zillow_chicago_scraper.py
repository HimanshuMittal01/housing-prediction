# import undetected_chromedriver as uc
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# import pandas as pd
# import time
# import random


# class ZillowScraper:
#     def __init__(self):
#         self.options = uc.ChromeOptions()

#         # Essential Chrome options
#         self.options.add_argument("--start-maximized")
#         self.options.add_argument("--disable-gpu")
#         self.options.add_argument("--no-sandbox")
#         self.options.add_argument("--disable-dev-shm-usage")

#         self.base_url = "https://www.zillow.com/chicago-il/"
#         self.properties = []
#         self.csv_file = "chicago_properties.csv"

#         # Create new CSV file with headers
#         pd.DataFrame(
#             columns=[
#                 "price",
#                 "address",
#                 "bedrooms",
#                 "bathrooms",
#                 "square_footage",
#                 "url",
#             ]
#         ).to_csv(self.csv_file, index=False)

#     def start_driver(self):
#         self.driver = uc.Chrome(options=self.options, version_main=131)
#         self.wait = WebDriverWait(self.driver, 20)
#         time.sleep(2)

#     def clean_property_data(self, property_info):
#         """Clean individual property data"""
#         cleaned_info = property_info.copy()

#         # Clean price
#         if cleaned_info["price"] != "N/A":
#             try:
#                 price_str = (
#                     cleaned_info["price"]
#                     .replace("$", "")
#                     .replace(",", "")
#                     .replace("+", "")
#                 )
#                 cleaned_info["price"] = float(price_str)
#             except:
#                 cleaned_info["price"] = pd.NA

#         # Clean numeric fields
#         for field in ["bedrooms", "bathrooms", "square_footage"]:
#             if cleaned_info[field] != "N/A" and cleaned_info[field] != "--":
#                 try:
#                     cleaned_info[field] = float(cleaned_info[field])
#                 except:
#                     cleaned_info[field] = pd.NA
#             else:
#                 cleaned_info[field] = pd.NA

#         return cleaned_info

#     def append_to_csv(self, property_info):
#         """Append a single property to CSV file"""
#         cleaned_info = self.clean_property_data(property_info)
#         pd.DataFrame([cleaned_info]).to_csv(
#             self.csv_file, mode="a", header=False, index=False
#         )

#     def extract_property_info(self):
#         try:
#             self.wait.until(
#                 EC.presence_of_element_located(
#                     (By.CSS_SELECTOR, 'div[id="grid-search-results"]')
#                 )
#             )
#             time.sleep(3)

#             property_cards = self.driver.find_elements(
#                 By.CSS_SELECTOR, 'article[class*="StyledPropertyCard"]'
#             )
#             print(f"Found {len(property_cards)} property cards")

#             for card in property_cards:
#                 try:
#                     # Extract price
#                     try:
#                         price = card.find_element(
#                             By.CSS_SELECTOR, '[data-test="property-card-price"]'
#                         ).text.strip()
#                     except:
#                         price = "N/A"

#                     # Extract address
#                     try:
#                         address = card.find_element(
#                             By.CSS_SELECTOR, '[data-test="property-card-addr"]'
#                         ).text.strip()
#                     except:
#                         address = "N/A"

#                     # Extract details with improved selector
#                     try:
#                         # Get the details list using the specific class
#                         details_list = card.find_element(
#                             By.CSS_SELECTOR,
#                             'ul[class*="StyledPropertyCardHomeDetailsList"]',
#                         )

#                         beds = baths = sqft = "N/A"

#                         # Get all list items
#                         detail_items = details_list.find_elements(
#                             By.TAG_NAME, "li"
#                         )

#                         # Print raw details for debugging
#                         print(
#                             "\nRaw details:",
#                             [item.text for item in detail_items],
#                         )

#                         for item in detail_items:
#                             # Get the number and unit separately
#                             try:
#                                 number = item.find_element(
#                                     By.TAG_NAME, "b"
#                                 ).text
#                                 unit = item.find_element(
#                                     By.TAG_NAME, "abbr"
#                                 ).text.lower()

#                                 print(
#                                     f"Processing: number={number}, unit={unit}"
#                                 )  # Debug print

#                                 if unit == "bds":
#                                     beds = number
#                                 elif unit == "ba":
#                                     baths = number
#                                 elif unit == "sqft":
#                                     sqft = number.replace(",", "")
#                             except Exception as e:
#                                 print(f"Error processing detail item: {e}")
#                                 continue

#                     except Exception as detail_error:
#                         print(f"Error extracting details: {detail_error}")
#                         beds = baths = sqft = "N/A"

#                     # Extract link
#                     try:
#                         link = card.find_element(
#                             By.CSS_SELECTOR, 'a[class*="property-card-link"]'
#                         ).get_attribute("href")
#                     except:
#                         link = "N/A"

#                     property_info = {
#                         "price": price,
#                         "address": address,
#                         "bedrooms": beds,
#                         "bathrooms": baths,
#                         "square_footage": sqft,
#                         "url": link,
#                     }

#                     # Store in memory
#                     self.properties.append(property_info)

#                     # Print extraction details
#                     print(f"Extracted: {address}")
#                     print(f"  Details: {beds} beds, {baths} baths, {sqft} sqft")
#                     print(f"  Price: {price}")
#                     print("-" * 50)

#                 except Exception as e:
#                     print(f"Error processing property card: {str(e)}")
#                     continue

#         except Exception as e:
#             print(f"Error during extraction: {str(e)}")
#             with open("error_page.html", "w", encoding="utf-8") as f:
#                 f.write(self.driver.page_source)

#     def scrape_properties(self, max_pages=3):
#         try:
#             self.start_driver()

#             for page in range(1, max_pages + 1):
#                 url = f"{self.base_url}?searchQueryState=%7B%22pagination%22%3A%7B%22currentPage%22%3A{page}%7D%7D"
#                 print(f"\nScraping page {page}")

#                 self.driver.get(url)
#                 time.sleep(random.uniform(5, 7))

#                 self.scroll_page()
#                 self.extract_property_info()
#                 print(f"Properties found so far: {len(self.properties)}")

#                 time.sleep(random.uniform(3, 5))

#             # Save all properties at once instead of appending
#             df = pd.DataFrame(self.properties)
#             df.to_csv(self.csv_file, index=False)
#             return df

#         finally:
#             if hasattr(self, "driver"):
#                 try:
#                     self.driver.quit()
#                 except:
#                     pass

#     def scroll_page(self):
#         """Scroll the page gradually"""
#         last_height = self.driver.execute_script(
#             "return document.body.scrollHeight"
#         )

#         while True:
#             # Scroll in smaller increments
#             for i in range(10):
#                 self.driver.execute_script(
#                     f"window.scrollTo(0, {(i + 1) * last_height / 10});"
#                 )
#                 time.sleep(random.uniform(0.7, 1.3))

#             time.sleep(2)
#             new_height = self.driver.execute_script(
#                 "return document.body.scrollHeight"
#             )
#             if new_height == last_height:
#                 break
#             last_height = new_height


# def main():
#     scraper = ZillowScraper()

#     try:
#         print("Starting scrape...")
#         df = scraper.scrape_properties(max_pages=20)

#         print("\nScraping completed!")
#         print(f"Total properties scraped: {len(df)}")
#         print("\nSample of scraped data:")
#         print(df.head())

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")


# if __name__ == "__main__":
#     main()
