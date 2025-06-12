# Public Safety Sound Classifier

A machine learning project that classifies short urban audio clips as **SAFE** or **UNSAFE** to assist in real-time public safety monitoring. This system was developed using the **UrbanSound8k** dataset and is deplyed as asimple **Flask web app* for audio file upload and classification.

This project showcases end-to-end ML developement: from data preprocessing and model training to building real-world interface for deployment. This interface is suitable for smart city applications and safety-critical environments. 

## Motivation: 
Urban sound detection can play a key role in safety systems for cities, campuses, or industrial sites. Detecting critical sounds such as **gunshots**, **rirens** , or **dog barks** helps flag potentially dangerous environments. This model aims to prioritze **recall for unsafe sounds**, ensuring that high risk events are rarely missed, even at the cost of  false alarms. 

## project Structure:

public-safety-sound-classifier/
├── app.py ← Flask web app
├── PublicSafetyClassifier.ipynb ← Full training notebook
├── models/
│ ├── safety_classifier_model.h5
│ └── label_encoder.pkl
├── templates/
│ ├── index.html
│ └── result.html
├── data/ ← Audio dataset 
│ └── UrbanSound8K.csv, folds
├── requirements.txt
├── README.md
└── .gitignore

## Model Overview: 


- **Dataset**: [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
- **Labels Used**:
  - **Unsafe**: `gun_shot`, `siren`, `dog_bark`
  - **Safe**: `children_playing`, `air_conditioner`, `drilling`
- **Feature Extraction**: 40-dimensional MFCCs using `librosa`
- **Classifier**: Feedforward neural network (Keras)
- **Evaluation Metrics**: Accuracy, Precision, Recall, Confusion Matrix
- **Recall Tuning**: Threshold shifting and class weighting to reduce false negatives for unsafe sounds


## Model Performance

After training on a filtered subset of UrbanSound8K:

- **Accuracy**: ~87%
- **Recall (Unsafe)**: **~86%** after threshold tuning and class weighting
- **False Negative Rate (Unsafe as Safe)**: Reduced from 100 to 65 with tuning

![Confusion Matrix](confusion_matrix_example.png)


## Web App: Flask Interface

You can upload a `.wav` file through the web interface, and the model will return:

- **SAFE** or **UNSAFE**
- Confidence score

### Example

1. User uploads `siren.wav`
2. Output: **UNSAFE (confidence: 0.93)**



## Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/yourusername/public-safety-sound-classifier.git
   cd public-safety-sound-classifier

2. Install dependencies: 
```bash 
pip install -r requirements.txt
```
3. Run the flask app:
```console 
python app.py
```
4. Visit http://127.0.0.1:5000

## Training Notebook: 
The full traning pipeline is available in the Jupyter notebook:
- MFCC extraction
- Class balancing (via weights)
- Threshold tunning (recall vs. precision trade-off)
- Evakuation metrics and plots