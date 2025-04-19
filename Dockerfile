FROM oogie0918/faiss:latest

WORKDIR /app
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install 'transformers[torch]==4.40.0'
RUN pip install open_clip_torch==2.23.0
RUN pip install pandas google-genai streamlit 
RUN pip install psycopg2-binary
RUN pip install python-dotenv boto3

ENV TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
RUN python -c "from transformers import AutoProcessor, AutoModel; \
    AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True); \
    AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)"


COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
