import time
import json
import os
import argparse
import logging
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("amicus_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

class AmicusScraper:
    def __init__(self, delay=2, save_interval=10, output_dir="output"):
        """
        Initialize the scraper with configurable parameters
        
        Args:
            delay: Delay between page loads in seconds
            save_interval: How often to save progress (in number of pages)
            output_dir: Directory to save the scraped data
        """
        self.delay = delay
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.driver = None
        self.amicus_data = []
        self.error_pages = []
        self.current_page = 0
        self.max_pages = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Check for existing progress file
        self.progress_file = os.path.join(output_dir, "progress.json")
        if os.path.exists(self.progress_file):
            self.load_progress()
    
    def start_browser(self):
        """Launch the browser for scraping"""
        logger.info("Starting browser...")
        
        # You may need to adjust options based on your environment
        options = webdriver.ChromeOptions()
        # Uncomment to run headless if needed
        # options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=options)
        logger.info("Browser started successfully")
        
    def load_progress(self):
        """Load saved progress if available"""
        try:
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.amicus_data = progress.get('data', [])
                self.error_pages = progress.get('error_pages', [])
                self.current_page = progress.get('current_page', 0)
                logger.info(f"Loaded progress: {len(self.amicus_data)} records, last page: {self.current_page}")
        except Exception as e:
            logger.error(f"Error loading progress: {e}")
    
    def save_progress(self):
        """Save current progress to file"""
        progress = {
            'data': self.amicus_data,
            'error_pages': self.error_pages,
            'current_page': self.current_page,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f)
            
            # Also save the full dataset separately
            dataset_file = os.path.join(self.output_dir, f"amicus_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(dataset_file, 'w') as f:
                json.dump(self.amicus_data, f)
                
            logger.info(f"Progress saved: {len(self.amicus_data)} records, last page: {self.current_page}")
        except Exception as e:
            logger.error(f"Error saving progress: {e}")
    
    def wait_for_user_login(self):
        """Pause execution until user indicates they've logged in"""
        input("\nPlease navigate to the database and log in.\nPress Enter when ready to start scraping...")
        logger.info("User indicated ready to proceed with scraping")
    
    def extract_page_data(self):
        """Extract amicus data from the current page"""
        try:
            # Wait for the page to load properly
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.ID, "docsGridContainer"))
            )
            
            # Extract case details
            case_data = {}
            
            # Case ID/Number
            try:
                case_number = self.driver.find_element(By.XPATH, "//div[contains(@class, 'segColL') and text()='Supreme Court case number']/following-sibling::div").text
                case_data['case_number'] = case_number
            except NoSuchElementException:
                case_data['case_number'] = None
                
            # Date filed
            try:
                date_filed = self.driver.find_element(By.XPATH, "//div[contains(@class, 'segColL') and text()='Date filed']/following-sibling::div").text
                case_data['date_filed'] = date_filed
            except NoSuchElementException:
                case_data['date_filed'] = None
                
            # Case name
            try:
                case_title = self.driver.find_element(By.XPATH, "//div[contains(@class, 'docSegRow')]/div[contains(@class, 'segColL') and text()='Case']/following-sibling::div//a").text
                case_data['case_title'] = case_title
            except NoSuchElementException:
                case_data['case_title'] = None
            
            # Position (Support of Petitioner/Respondent)
            try:
                position = self.driver.find_element(By.XPATH, "//div[contains(@class, 'segColL') and text()='Position']/following-sibling::div").text
                case_data['position'] = position
            except NoSuchElementException:
                case_data['position'] = None
            
            # Extracting Amicus Curiae organizations
            amicus_orgs = []
            try:
                # Find all amicus organizations listed on the page
                amicus_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'segColL') and text()='Amicus curiae']/following-sibling::div | //div[contains(@class, 'segColL') and text()='']/following-sibling::div[contains(@class, 'segColR')]//a[contains(@doctype, 'sci:amicus-normalized')]")
                
                for element in amicus_elements:
                    amicus_name = element.text.strip()
                    if amicus_name and not amicus_name.startswith("Show all"):
                        amicus_orgs.append(amicus_name)
            except Exception as e:
                logger.error(f"Error extracting amicus organizations: {e}")
            
            case_data['amicus_organizations'] = amicus_orgs
            
            # Petitioners
            petitioners = []
            try:
                petitioner_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'segColL') and text()='Petitioner']/following-sibling::div | //div[contains(@class, 'segColL') and text()='']/following-sibling::div[contains(@class, 'segColR')]//a[contains(@doctype, 'sci:petitioner-normalized')]")
                
                for element in petitioner_elements:
                    petitioner_name = element.text.strip()
                    if petitioner_name and not petitioner_name.startswith("Show all"):
                        petitioners.append(petitioner_name)
            except Exception as e:
                logger.error(f"Error extracting petitioners: {e}")
                
            case_data['petitioners'] = petitioners
            
            # Respondents
            respondents = []
            try:
                respondent_elements = self.driver.find_elements(By.XPATH, "//div[contains(@class, 'segColL') and text()='Respondent']/following-sibling::div | //div[contains(@class, 'segColL') and text()='']/following-sibling::div[contains(@class, 'segColR')]//a[contains(@doctype, 'sci:respondent-normalized')]")
                
                for element in respondent_elements:
                    respondent_name = element.text.strip()
                    if respondent_name and not respondent_name.startswith("Show all"):
                        respondents.append(respondent_name)
            except Exception as e:
                logger.error(f"Error extracting respondents: {e}")
                
            case_data['respondents'] = respondents
            
            # Get current URL for reference
            case_data['url'] = self.driver.current_url
            
            # Add to our dataset
            self.amicus_data.append(case_data)
            logger.info(f"Successfully extracted data for case {case_data.get('case_number')}")
            
            return True
            
        except TimeoutException:
            logger.error(f"Timeout waiting for page elements on page {self.current_page}")
            self.error_pages.append({
                'page': self.current_page,
                'url': self.driver.current_url,
                'error': 'Timeout waiting for page elements'
            })
            return False
        except Exception as e:
            logger.error(f"Error extracting data from page {self.current_page}: {e}")
            self.error_pages.append({
                'page': self.current_page,
                'url': self.driver.current_url,
                'error': str(e)
            })
            return False
    
    def go_to_next_page(self):
        """Navigate to the next page in the results"""
        try:
            # Look for the next button and click it
            next_button = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@title, 'next document')]"))
            )
            
            # Extract current page and total pages info if available
            pagination_info = self.driver.find_element(By.XPATH, "//span[contains(@id, 'docviewprevnext')]/strong").text.strip()
            total_pages_text = self.driver.find_element(By.XPATH, "//span[contains(@id, 'docviewprevnext')]/span").text.strip()
            
            logger.info(f"Current position: {pagination_info} {total_pages_text}")
            
            # Extract total pages if not already known
            if self.max_pages is None and "of " in total_pages_text:
                try:
                    self.max_pages = int(total_pages_text.split("of ")[1].strip().split()[0].replace(",", ""))
                    logger.info(f"Total pages detected: {self.max_pages}")
                except:
                    logger.warning("Could not determine total pages")
            
            # Click the next button
            next_button.click()
            
            # Wait for the page to load
            time.sleep(self.delay)
            
            self.current_page += 1
            return True
        except TimeoutException:
            logger.error("Timeout finding next button - may have reached the end")
            return False
        except Exception as e:
            logger.error(f"Error navigating to next page: {e}")
            return False
    
    def run_scraper(self, limit=None):
        """
        Run the scraping process
        
        Args:
            limit: Maximum number of pages to scrape (None for unlimited)
        """
        try:
            # Wait for user to login and start the process
            self.wait_for_user_login()
            
            # Extract initial page data
            success = self.extract_page_data()
            if success:
                logger.info(f"Scraped page {self.current_page}")
            
            # Set the limit if provided
            page_limit = limit if limit is not None else float('inf')
            
            # Main scraping loop
            while (self.current_page < page_limit and 
                  (self.max_pages is None or self.current_page < self.max_pages)):
                
                # Go to next page
                next_page_success = self.go_to_next_page()
                if not next_page_success:
                    logger.info("Could not navigate to next page, ending scraping")
                    break
                
                # Extract data from the current page
                success = self.extract_page_data()
                if success:
                    logger.info(f"Scraped page {self.current_page}")
                
                # Save progress at intervals
                if self.current_page % self.save_interval == 0:
                    self.save_progress()
                
                # Check if user wants to pause
                if self.current_page % 20 == 0:
                    continue_scraping = input(f"\nContinue scraping? Currently on page {self.current_page}. (y/n): ")
                    if continue_scraping.lower() != 'y':
                        logger.info("User requested to stop scraping")
                        break
            
            # Final save
            self.save_progress()
            
            logger.info(f"Scraping completed. Total records: {len(self.amicus_data)}")
            logger.info(f"Error pages: {len(self.error_pages)}")
            
        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")
            self.save_progress()
        except Exception as e:
            logger.error(f"Unexpected error during scraping: {e}")
            self.save_progress()
        finally:
            if self.driver:
                self.driver.quit()

def main():
    """Main entry point with command line argument handling"""
    parser = argparse.ArgumentParser(description="Supreme Court Amicus Brief Scraper")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between page loads in seconds")
    parser.add_argument("--save-interval", type=int, default=10, help="How often to save progress (in number of pages)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of pages to scrape")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save the scraped data")
    
    args = parser.parse_args()

    print(f"Starting scraper with parameters: {args}")
    
    # Initialize and run the scraper
    scraper = AmicusScraper(
        delay=args.delay,
        save_interval=args.save_interval,
        output_dir=args.output_dir
    )
    
    scraper.start_browser()
    scraper.run_scraper(limit=args.limit)

if __name__ == "__main__":
    main()