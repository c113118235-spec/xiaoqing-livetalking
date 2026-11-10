# IndexTTS-2.5 服務：借用 xiaozhi-tts2 已經能在 DGX Spark（ARM + 新顯卡）跑的 torch，只補 IndexTTS 需要的套件
# 套件盡量 --no-deps，避免 pip 把顯卡版 torch 換掉
FROM xiaozhi-tts2
RUN pip install --no-cache-dir fastapi uvicorn opencc-python-reimplemented cn2an jieba json5 munch omegaconf \
      textstat g2p-en fugashi unidic-lite ffmpeg-python sentencepiece einops safetensors \
 && pip install --no-cache-dir --no-deps descript-audio-codec descript-audiotools \
 && pip install --no-cache-dir --no-deps argbind julius flatten-dict pyloudnorm markdown2 ffmpy randomname \
      importlib-resources docstring-parser fire \
 && python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
WORKDIR /idx
ENTRYPOINT []
CMD ["python", "/srv/indextts_server.py"]
