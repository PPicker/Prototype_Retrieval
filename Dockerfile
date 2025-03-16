FROM python:3.10-slim

WORKDIR /app
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install 'transformers[torch]'
RUN pip install open_clip_torch
RUN pip install pandas numpy google-genai streamlit 


RUN python -c "from transformers import AutoProcessor, AutoModel; \
    AutoProcessor.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True); \
    AutoModel.from_pretrained('Marqo/marqo-fashionCLIP', trust_remote_code=True)"


COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
