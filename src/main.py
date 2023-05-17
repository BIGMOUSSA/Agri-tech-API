from fastapi import FastAPI
import uvicorn
from typing import List, Literal
from pydantic import BaseModel
import pandas as pd
import pickle, os
from fastapi.middleware.cors import CORSMiddleware
import heapq


# Useful functions
def load_ml_components(fp):
    "Load the ml components to re-use in app"
    with open(fp, "rb") as f:
        object = pickle.load(f)
    return object


# INPUT MODELING
class Land(BaseModel):
    """Modeling of one input data in a type-restricted dictionary-like format

    column_name : variable type # strictly respect the name in the dataframe header.

    eg.:
    =========
    customer_age : int
    gender : Literal['male', 'female', 'other']
    """

    ## Input features
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


# Setup
## variables and constants
DIRPATH = os.path.dirname(os.path.realpath(__file__))
ml_core_fp = os.path.join(DIRPATH, "assets", "ml", "crop_recommandation2.pkl")

## Loading
ml_components_dict = load_ml_components(fp=ml_core_fp)
##### Avant de charger tu dois sauvegarder le dictionnaire
# {
#     "model": model,
#     "scaler": scaler,
#     "labels": labels,
# }

labels = ml_components_dict["labels"]
idx_to_labels = {i: l for (i, l) in enumerate(labels)}

model = ml_components_dict["model"]
##### Tu peux mettre ton scaler sur la ligne suivante
scaler = ml_components_dict["scaler"]

print(f"\n[Info] Predictable labels: {labels}")
print(f"\n[Info] Indexes to labels: {idx_to_labels}")
print(f"\n[Info] ML components loaded: {list(ml_components_dict.keys())}")

# API
app = FastAPI(title="Land classification API")

# Configure CORS
origins = ["*"]  # Update with your allowed origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "info": "Crop and land matching prediction API : This is my API to predict crops regarding some features."
    }


@app.post("/predict")
async def predict(land: Land):
    try:
        # Dataframe creation
        df = pd.DataFrame(
            {
                "N": [land.N],
                "P": [land.P],
                "K": [land.K],
                "temperature": [land.temperature],
                "humidity": [land.humidity],
                "ph": [land.ph],
                "rainfall": [land.rainfall],
            }
        )
        print(f"[Info] Input data as dataframe :\n{df}")
        # df.columns = num_cols+cat_cols
        # df = df[num_cols+cat_cols] # reorder the dateframe

        # ML part
        ##### Tu peux mettre ton scaler sur la ligne suivante
        scaler = ml_components_dict["scaler"]
        df = scaler.transform(df)
        data_ready = df
        output = model.predict_proba(data_ready)
        
        print(output)
        print('uiytretttttttttttttttttttttttt')
        output = list(output[0])
        confidence_score_max = output.index(max(output))
        confidence_score = max(output)
        predicted_label = labels[confidence_score_max]
        print(confidence_score)
        print('uiytretttttttttttttttttttttttt')
        ## store confidence score/ probability for the predicted class
        #confidence_score = output.max(axis=-1)
        #print(confidence_score[0])
        #df["confidence score"] = confidence_score[0]
        
        #print('tttttttttttttttttttttttttà')
        ## get index of the predicted class
        #predicted_idx = output.argmax(axis=-1)

        # store index then replace by the matching label
        #df["predicted label"] = predicted_idx
        #predicted_label = df["predicted label"].replace(idx_to_labels)
        #df["predicted label"] = predicted_label
        print(
            f"✅ The best crop for this land is : '{predicted_label}' with a confidence score of '{confidence_score}' .",
        )
        msg = "Execution went fine"
        code = 200
        
        pred = {
            "confidence_score" : confidence_score.item(),
            "predicted_label" : predicted_label
        }
    except Exception as e:
        print(e)
        print(f"🚨 Something went wrong during the prediction.")
        msg = "Execution went wrong"
        code = 0
        pred = None
    
    result = {"execution_msg": msg, "execution_code": code, "predictions": pred}
    return result
    


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
