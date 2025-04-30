from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import numpy as np
import time
from typing import Dict


class Embedder():
    def __init__(self):
        self.tmp = 0
        self.api_keys = [os.getenv("GEMINI_API_KEY"),
                    os.getenv("GEMINI_API_KEY2"),
                    os.getenv("GEMINI_API_KEY3"),
                    os.getenv("GEMINI_API_KEY4"),
                    os.getenv("GEMINI_API_KEY5"),
                    ]
        self.client = genai.Client(
                api_key=self.api_keys[self.tmp],
            )
    def embed(self, text_query):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = self.client.models.embed_content(
                        model="gemini-embedding-exp-03-07",
                        contents=text_query,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                embedding_np = np.array(result.embeddings[0].values, dtype=np.float32)
                emb_normalized = embedding_np / np.linalg.norm(embedding_np)
                return emb_normalized
                
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                
                self.tmp = (self.tmp+1) % 5
                self.client = genai.Client(
                    api_key=self.api_keys[self.tmp],
                )
        return None