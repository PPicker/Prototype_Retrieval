FROM oogie0918/faiss:latest

WORKDIR /app
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install 'transformers[torch]==4.42.4'
RUN pip install open_clip_torch==2.23.0
RUN pip install pandas google-genai streamlit 
RUN pip install psycopg2-binary pgvector
RUN pip install python-dotenv boto3

ENV TORCH_FORCE_WEIGHTS_ONLY_LOAD=1
RUN python -c "from transformers import CLIPProcessor, CLIPModel; \
    CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); \
    CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')"


COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
