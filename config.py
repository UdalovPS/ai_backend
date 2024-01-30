"""module with global configure"""

from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH_1 = os.getenv("MODEL_PATH_1")
MODEL_PATH_2 = os.getenv("MODEL_PATH_2")

# mkl-fft
# mkl-random
# mkl-service==2.4.0
