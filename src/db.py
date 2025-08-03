# src/db.py

from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB URI from .env
MONGO_URI = os.getenv("MONGO_URI")

# Ensure the URI is present
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables. Check your .env file.")

# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)
db = client["fruit_quality"]
collection = db["uploads"]

def save_upload_metadata(filename, label, filepath):
    """Save metadata of uploaded image to MongoDB."""
    doc = {
        "filename": filename,
        "label": label,
        "filepath": filepath,
        "upload_time": datetime.utcnow()
    }
    collection.insert_one(doc)

def get_all_uploads():
    """Retrieve all uploaded entries from MongoDB (excluding _id)."""
    return list(collection.find({}, {"_id": 0}))
