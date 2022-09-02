from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model, load_model
import numpy as np
from PIL import Image
import json

app = FastAPI()

# https://github.com/luchonaveiro/image-search-engine thanks to!
@app.post("/similarity/")
async def create_upload_files(files: list[UploadFile], files2: list[UploadFile]):
    print('inputfile-> ', files2[0].filename)
    for file in files:
        print('comparefile-> ', file.filename)

    # 유클리디안 거리
    def eucledian_distance(x,y):
        eucl_dist = np.linalg.norm(x - y)
        return eucl_dist
    rst = []

    # 모델로드
    autoencoder = load_model('D:/fastapi/backend/app/image_autoencoder_2.h5', compile=False)
    latent_space_model = Model(autoencoder.input, autoencoder.get_layer('latent_space').output)
    
    input_img = files2[0]
    input_img = await input_img.read()
    input_img = Image.open(BytesIO(input_img))
    input_img = input_img.resize((256, 256))
    input_img = img_to_array(input_img) / 255.0
    input_img = np.expand_dims(input_img, axis=0)
    input_pred = latent_space_model.predict(input_img)
    rst.append(input_pred)

    for file in files:
        file = await file.read()
        file = Image.open(BytesIO(file))
        file = file.resize((256, 256))
        file = img_to_array(file) / 255.0
        file = np.expand_dims(file, axis=0)
        file = latent_space_model.predict(file)
        rst.append(file)

    rst_list = []
    for i in range(1, len(rst)):
        rst_list.append(eucledian_distance(rst[i], rst[0]))

    print(type(rst_list[0]))
    return {'similarity': [np.float64(v).item() for v in rst_list]}

@app.get("/")
async def main():
    content = """
<body>
<form action="/similarity/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input name="files2" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)