# rag_pdf_chat_sreekar
## Overview
The goal of this project is to implement a Retrieval-Augmented Generation (RAG) pipeline to allow users to interact with semi-structured data in PDF files. This includes extracting, chunking, embedding, and querying data efficiently using vector-based retrieval.

## Project Structure
```
rag_pdf_chat_sreekar/
├── data/
│   └── sample_pdfs/
│       └── sethafal.pdf
├── src/
│   ├── pipeline/
│   │   └── rag_pipeline.py
│   └── utils/
|   |   └── helpers.py
│   └── interface/
|   |   └── templates
|   |   |   └── index.html
├── README.md
├── requirements.txt
```

## Setup Instructions
1. Clone the repository:
```bash
git clone https://github.com/sreekar8811/rag_pdf_chat_sreekar.git
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Add your OpenAI API Key:
- Create a `.env` file with `OPENAI_API_KEY="open_api_secert_key_from_https://platform.openai.com/settings/organization/api-keys"`.

4. Run the pipeline script for example usage or integrate it with the app.
