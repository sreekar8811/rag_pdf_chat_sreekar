import os
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pandas as pd
import tabula
import re

class PDFRagPipeline:
    def __init__(self, openai_api_key):
        """Initialize the RAG pipeline with the required components."""
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_text_and_tables(self, pdf_path):
        """Extract text and tables from a PDF file."""
        try:
            # Extract text
            pdf_reader = pypdf.PdfReader(pdf_path)
            text = "".join(page.extract_text() for page in pdf_reader.pages)

            # Extract tables
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
            tables_text = "\n".join([table.to_string(index=False) for table in tables])

            return text, tables_text
        except Exception as e:
            raise ValueError(f"Error extracting content from {pdf_path}: {e}")

    def process_pdf(self, pdf_path):
        """Process PDF content and store embeddings in a vector database."""
        text, tables_text = self.extract_text_and_tables(pdf_path)
        combined_text = text + "\n" + tables_text

        # Split text into chunks
        chunks = self.text_splitter.split_text(combined_text)

        # Create or update the vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)

    def query(self, user_query, top_k=4):
        """Handle a user query and return a response."""
        if self.vector_store is None:
            return "No data processed. Please upload and process PDFs first."

        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=retriever
        )
        return qa_chain.run(user_query)

    def compare_data(self, comparison_query, top_k=6):
        """Handle a comparison query and return structured output."""
        docs = self.vector_store.similarity_search(comparison_query, k=top_k)
        context = "\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Based on the following context, compare the requested information:
        Context: {context}
        Query: {comparison_query}
        Provide a clear and structured comparison with exact values.
        """

        response = self.llm.generate([prompt])
        return response.generations[0][0].text

    def extract_unemployment_data(self, text):
        """Extract unemployment rates by degree type."""
        pattern = r"(\w+\s*\w*\s*degree).*?(\d+\.\d+)%"
        matches = re.findall(pattern, text, re.IGNORECASE)
        return {degree.strip(): float(rate) for degree, rate in matches}

    def extract_specific_table(self, pdf_path, page_number):
        """Extract specific table data from a given PDF page."""
        try:
            tables = tabula.read_pdf(pdf_path, pages=page_number, multiple_tables=False, silent=True)
            return tables[0] if tables else None
        except Exception as e:
            raise ValueError(f"Error extracting table from page {page_number}: {e}")

# Example usage:
if __name__ == "__main__":
    pipeline = PDFRagPipeline(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Process a PDF
    pdf_path = "data/sample_pdfs/sethafal.pdf"
    pipeline.process_pdf(pdf_path)

    # Handle a general query
    query_result = pipeline.query("What is the unemployment rate for bachelor's degrees?")
    print("Query Result:", query_result)

    # Handle a comparison query
    comparison_result = pipeline.compare_data("Compare unemployment rates for bachelor's and master's degrees.")
    print("Comparison Result:", comparison_result)
