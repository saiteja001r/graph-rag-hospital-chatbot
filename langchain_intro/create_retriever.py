import dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

REVIEW_CSV = r"C:\Users\dhira\Desktop\ALL_Projects\Bunny_proj\Graph_RAG\data\reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

dotenv.load_dotenv()

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )



loader = CSVLoader(file_path=REVIEW_CSV, source_column="review")
reviews = loader.load() 

# text = text_splitter.create_documents(reviews)

reviews_vector_db = Chroma.from_documents(
    documents=reviews,
    embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory=REVIEWS_CHROMA_PATH,
)

