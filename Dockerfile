# Version: 2.0.0
FROM paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82

# PaddleOCR base on Python3.7
RUN pip3.7 install --upgrade pip -i https://mirror.baidu.com/pypi/simple
# Install Paddle
RUN pip3.7 install paddlepaddle-gpu==2.1.1.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

RUN mkdir -p /PaddleOCR/
# Copy code
COPY ["./PaddleOCR",  "/PaddleOCR/"]

RUN cd /PaddleOCR && pip3.7 install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

RUN mkdir -p /PaddleOCR/inference/
# Copy orc recognition model(light version). If you want to change normal version, you can change ch_ppocr_mobile_v2.0_rec_infer to ch_ppocr_server_v2.0_rec_infer, also remember change rec_model_dir in deploy/hubserving/ocr_system/params.py）
COPY ["./inference/inference.*",  "/PaddleOCR/inference/"]





