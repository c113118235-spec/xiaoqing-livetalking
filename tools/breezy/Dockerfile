# BreezyVoice 服務：借用 xiaozhi-tts2（CosyVoice3）已經能在 DGX Spark 跑的環境，只補幾個小套件
FROM xiaozhi-tts2
RUN pip install --no-cache-dir opencc-python-reimplemented fastapi uvicorn \
 && pip install --no-cache-dir --no-deps g2pw
WORKDIR /bv
ENTRYPOINT []
CMD ["python", "/srv/breezy_server.py"]
