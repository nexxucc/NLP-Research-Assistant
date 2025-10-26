"""
Download Papers from Semantic Scholar API

Downloads research papers on a specific topic using Semantic Scholar API.
Filters for papers with open-access PDFs.

Usage:
    python scripts/download_papers_semantic_scholar.py --query "drones" --num-papers 1000

Get free API key: https://www.semanticscholar.org/product/api#api-key-form
"""

import requests
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def search_papers(query, num_papers=1000, api_key=None, year_range=None, field=None):
    """
    Search for papers on Semantic Scholar.

    Args:
        query: Search query (e.g., "drones computer vision")
        num_papers: Number of papers to find
        api_key: Semantic Scholar API key (optional but recommended)
        year_range: Year filter (e.g., "2020-2024")
        field: Field filter (e.g., "Computer Science")

    Returns:
        List of paper dictionaries
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
        print(f"Using API key: {api_key[:8]}...")
    else:
        print("No API key provided. Using public rate limits (slower).")

    all_papers = []
    offset = 0
    limit = 100  # Max per request

    # API limit: offset + limit must be < 1000
    max_api_results = 1000

    print(f"\nSearching for papers on: '{query}'")
    if year_range:
        print(f"Year range: {year_range}")
    if field:
        print(f"Field: {field}")

    # Cap num_papers to API limit
    if num_papers > max_api_results:
        print(f"⚠️  Note: API limit is {max_api_results} papers. Capping to {max_api_results}.")
        num_papers = max_api_results

    with tqdm(total=num_papers, desc="Searching papers") as pbar:
        while len(all_papers) < num_papers and offset < max_api_results:
            params = {
                "query": query,
                "limit": limit,
                "offset": offset,
                "fields": "paperId,title,year,authors,abstract,openAccessPdf,citationCount,venue,publicationDate"
            }

            if year_range:
                params["year"] = year_range
            if field:
                params["fieldsOfStudy"] = field

            try:
                response = requests.get(base_url, params=params, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])

                    if not papers:
                        print(f"\nNo more papers found. Total: {len(all_papers)}")
                        break

                    # Filter for papers with valid open access PDF URLs
                    papers_with_pdf = []
                    for p in papers:
                        pdf_data = p.get("openAccessPdf")
                        if pdf_data and pdf_data.get("url") and pdf_data["url"].startswith("http"):
                            papers_with_pdf.append(p)

                    all_papers.extend(papers_with_pdf)

                    pbar.update(len(papers_with_pdf))
                    offset += limit

                    # Rate limiting (be nice to the API)
                    if not api_key:
                        time.sleep(1)  # 1 second between requests without key
                    else:
                        time.sleep(0.1)  # 100ms with key

                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 60))
                    print(f"\nRate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)

                elif response.status_code == 400:
                    # API limit reached
                    print(f"\n⚠️  Reached API limit (offset {offset}). Stopping search.")
                    print(f"   Collected {len(all_papers)} papers with valid PDFs.")
                    break

                else:
                    print(f"\nError: {response.status_code} - {response.text}")
                    break

            except Exception as e:
                print(f"\nError searching papers: {e}")
                break

    print(f"\n✓ Found {len(all_papers)} papers with open-access PDFs")
    return all_papers[:num_papers]


def download_pdf(url, filepath, timeout=20, max_retries=2):
    """Download a PDF from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                stream=True,
                headers={'User-Agent': 'Mozilla/5.0'}  # Some servers require this
            )
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            elif response.status_code == 403:
                # Forbidden - don't retry
                return False
            elif attempt < max_retries - 1:
                # Retry on other errors
                time.sleep(2)
                continue
            return False
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException) as e:
            # Network errors - silently skip (too many to log)
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False
        except Exception:
            return False
    return False


def download_papers(papers, output_dir="data/papers"):
    """
    Download PDFs for the papers.

    Args:
        papers: List of paper dictionaries from search
        output_dir: Where to save PDFs
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(papers)} papers to {output_dir}/")

    downloaded = 0
    failed = 0
    skipped = 0

    for i, paper in enumerate(tqdm(papers, desc="Downloading PDFs")):
        # Validate PDF URL exists and is valid
        if not paper.get('openAccessPdf'):
            skipped += 1
            continue

        pdf_url = paper['openAccessPdf'].get('url', '')

        # Skip if URL is empty or invalid
        if not pdf_url or not pdf_url.startswith('http'):
            skipped += 1
            continue

        # Create safe filename
        paper_id = paper.get('paperId', f'paper_{i}')
        year = paper.get('year', 'unknown')

        # Sanitize title for filename
        title = paper.get('title', 'untitled')
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title[:50]  # Limit length

        filename = f"{year}_{paper_id}_{safe_title}.pdf"
        filepath = output_path / filename

        # Skip if already exists
        if filepath.exists():
            downloaded += 1
            continue

        # Download PDF
        if download_pdf(pdf_url, filepath):
            downloaded += 1
        else:
            failed += 1

        # Be nice to servers (but not too slow)
        time.sleep(0.2)

    print(f"\n✓ Downloaded: {downloaded}")
    print(f"✗ Failed: {failed}")
    print(f"⊘ Skipped (no valid PDF URL): {skipped}")
    print(f"\nPapers saved to: {output_path.absolute()}")


def save_metadata(papers, output_file="data/papers_metadata.json"):
    """Save paper metadata to JSON file."""
    import json

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"✓ Metadata saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download research papers from Semantic Scholar")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g., 'drones computer vision')")
    parser.add_argument("--num-papers", type=int, default=1000, help="Number of papers to download")
    parser.add_argument("--output-dir", type=str, default="data/papers", help="Output directory for PDFs")
    parser.add_argument("--year", type=str, help="Year range (e.g., '2020-2024')")
    parser.add_argument("--field", type=str, help="Field of study (e.g., 'Computer Science')")
    parser.add_argument("--api-key", type=str, help="Semantic Scholar API key (or set SEMANTIC_SCHOLAR_API_KEY env var)")

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.getenv('SEMANTIC_SCHOLAR_API_KEY')

    print("=" * 80)
    print("SEMANTIC SCHOLAR PAPER DOWNLOADER")
    print("=" * 80)

    # Search for papers
    papers = search_papers(
        query=args.query,
        num_papers=args.num_papers,
        api_key=api_key,
        year_range=args.year,
        field=args.field
    )

    if not papers:
        print("\n✗ No papers found. Try a different query.")
        return

    # Save metadata
    save_metadata(papers, f"{args.output_dir}_metadata.json")

    # Download PDFs
    download_papers(papers, args.output_dir)

    print("\n" + "=" * 80)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Check papers in: {args.output_dir}/")
    print(f"2. Run: python scripts/build_index.py")
    print(f"3. Start querying your papers!")


if __name__ == "__main__":
    main()
