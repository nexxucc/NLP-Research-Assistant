"""
ArXiv Paper Scraper - No API, No Limits

Scrapes papers directly from arXiv.org using web scraping.
Downloads unlimited papers on any topic.

Usage:
    python scripts/scrape_arxiv_papers.py --query "drones" --num-papers 5000
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import re

def search_arxiv_web(query, num_papers=5000):
    """
    Scrape arXiv search results directly from the website.
    NO API, NO LIMITS.

    Args:
        query: Search term
        num_papers: How many papers to find

    Returns:
        List of paper URLs
    """
    papers = []
    start = 0
    page_size = 200  # ArXiv shows up to 200 results per page

    print(f"\nScraping arXiv for '{query}'...")
    print(f"Target: {num_papers} papers\n")

    with tqdm(total=num_papers, desc="Finding papers") as pbar:
        while len(papers) < num_papers:
            # ArXiv search URL
            url = f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first&size={page_size}&start={start}"

            try:
                response = requests.get(url, timeout=30)

                if response.status_code != 200:
                    print(f"\nError: Status {response.status_code}")
                    break

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Find all paper links
                paper_links = soup.find_all('p', class_='list-title')

                if not paper_links:
                    print(f"\nNo more papers found at start={start}")
                    break

                for link in paper_links:
                    a_tag = link.find('a')
                    if a_tag and 'href' in a_tag.attrs:
                        paper_url = a_tag['href']
                        # Convert to PDF URL
                        paper_id = paper_url.split('/')[-1]
                        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
                        papers.append({
                            'url': pdf_url,
                            'id': paper_id,
                            'abstract_url': f"https://arxiv.org{paper_url}"
                        })
                        pbar.update(1)

                        if len(papers) >= num_papers:
                            break

                start += page_size
                time.sleep(1)  # Be polite to arXiv

            except Exception as e:
                print(f"\nError scraping page: {e}")
                break

    print(f"\n✓ Found {len(papers)} papers")
    return papers


def download_paper(paper, output_dir):
    """Download a single paper."""
    pdf_url = paper['url']
    paper_id = paper['id']

    # Filename
    filename = f"{paper_id}.pdf"
    filepath = Path(output_dir) / filename

    # Skip if exists
    if filepath.exists():
        return True

    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            stream=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        return False

    except Exception:
        return False


def download_all_papers(papers, output_dir="data/papers"):
    """Download all papers."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(papers)} papers to {output_dir}/\n")

    success = 0
    failed = 0

    for paper in tqdm(papers, desc="Downloading PDFs"):
        if download_paper(paper, output_dir):
            success += 1
        else:
            failed += 1

        time.sleep(0.5)  # Be polite

    print(f"\n✓ Downloaded: {success}")
    print(f"✗ Failed: {failed}")
    print(f"\nPapers saved to: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="Scrape papers from arXiv (no API, no limits)")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--num-papers", type=int, default=1000, help="Number of papers")
    parser.add_argument("--output-dir", type=str, default="data/papers", help="Output directory")

    args = parser.parse_args()

    print("=" * 80)
    print("ARXIV WEB SCRAPER - No API, No Limits")
    print("=" * 80)

    # Scrape papers
    papers = search_arxiv_web(args.query, args.num_papers)

    if not papers:
        print("\n✗ No papers found")
        return

    # Download
    download_all_papers(papers, args.output_dir)

    print("\n" + "=" * 80)
    print("✓ SCRAPING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
