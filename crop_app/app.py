import torch, torchvision
import mmcls
from PIL import Image
import mmcv
from mmcls.apis import inference_model, init_model, show_result_pyplot
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from io import BytesIO
import numpy as np

app = FastAPI()

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/crop_mmclassification/")
async def crop_mmclassification(input: UploadFile = File()):
    config_file = 'configs/custom_config.py'
    checkpoint_file = '/data/mm/mmclassification/work_dirs/Resnet50/epoch_50.pth'
    device = 'cuda:0'
    model = init_model(config_file, checkpoint_file, device=device)
    
    print('inputfile-> ', input.filename)
    print('inputfiletype-> ', type(input))
    
    image = load_image_into_numpy_array(await input.read())

    img_array =  mmcv.imread(image)
    print(img_array)
    result = inference_model(model, img_array)
    print(result)
    
@app.get("/")
async def main():
    content = """
<body>
<form action="/crop_mmclassification/" enctype="multipart/form-data" method="post">
<input name="input" type="file">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)