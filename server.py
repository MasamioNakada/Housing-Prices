import pandas as pd
# Pydantic
from pydantic import BaseModel
from pydantic import Field

# FastAPI
from fastapi import FastAPI
from fastapi import status
from fastapi import Body, Query,Path,UploadFile,File

app = FastAPI()


#Models

@app.get(
    path='/',
    summary="Home",
    tags=['Home']
)
def home():
    return {'message':'To see documentation -> localhost/docs'}

@app.post(
    path='/prediction',
    summary="prediction",
    tags=['prediction']
)
def prediction(
    to_pred : UploadFile = File(...)
):
    if to_pred.content_type == 'text/csv':
        y_pred = pd.read_csv('./out/pred_test.csv')
        return {
            "Filename": to_pred.filename,
            "Format": to_pred.content_type,
            "Size(kb)": f'{round(len(to_pred.file.read())/1024, ndigits=2)} kb',
            "Prediction": list(y_pred['pred'])
        }

    else:
        return status.HTTP_400_BAD_REQUEST
