import arxiv
import os

def download_papers(topic="Retrieval Augmented Generation", limit=10, data_dir="data/pdfs"):
    """
    Downloads top research papers from ArXiv to populate the Knowledge Base.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Search ArXiv
    print(f"Searching ArXiv for top {limit} papers on '{topic}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query = topic,
        max_results = limit,
        sort_by = arxiv.SortCriterion.SubmittedDate
    )

    count = 0
    for result in client.results(search):
        # Create a safe filename
        safe_title = "".join(x for x in result.title if x.isalnum() or x in " -_").strip()
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            result.download_pdf(dirpath=data_dir, filename=filename)
            print(f"✅ Downloaded: {filename}")
            count += 1
        else:
            print(f"⚠️ Skipped (Exists): {filename}")

    print(f"\n🎉 Success! {count} new papers downloaded to '{data_dir}'.")
    print("Run your Streamlit app and click 'Load Knowledge Base' to process them.")

if __name__ == "__main__":
    # You can change the topic here to "Large Language Models" or "Transformers"
    download_papers(topic="RAG LLM", limit=15)