# Liver Disease Prediction App

A machine learning-powered web application built with Streamlit that predicts the likelihood of liver disease based on various health parameters.

## Features

- Interactive web interface for entering patient medical parameters
- Real-time liver disease risk prediction
- Probability scores for prediction confidence
- User-friendly input validation and error handling
- Responsive design with organized input fields

## Prerequisites

- Python 3.7 or higher
- Required model files (see Setup section)

## Installation

### 1. Clone or Download the Repository

```bash
git clone <your-repository-url>
cd liver-disease-prediction
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv liver_disease_env

# Activate virtual environment
# On Windows:
liver_disease_env\Scripts\activate
# On macOS/Linux:
source liver_disease_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Required Files

Before running the application, ensure you have the following files in your project directory:

1. **`best_liver_prediction_model.pkl`** - Your trained machine learning model
2. **`scaler.pkl`** - The fitted scaler used during model training
3. **`model_features.pkl`** - List of feature names in the correct order
4. **`streamlit_app.py`** - The main application file
5. **`requirements.txt`** - Python dependencies

### Generating Required Model Files

If you don't have the required `.pkl` files, you need to train your model first and save these components:

```python
import joblib

# After training your model
joblib.dump(trained_model, 'best_liver_prediction_model.pkl')
joblib.dump(fitted_scaler, 'scaler.pkl')
joblib.dump(list_of_feature_names, 'model_features.pkl')
```

## Running the Application

### Local Development

```bash
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501`

### Production Deployment

#### Option 1: Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path to `streamlit_app.py`
7. Click "Deploy"

#### Option 2: Heroku

1. Create a `Procfile` in your project root:
```
web: sh setup.sh && streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

#### Option 3: Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t liver-disease-app .
docker run -p 8501:8501 liver-disease-app
```

## Usage

1. Open the application in your web browser
2. Enter the patient's medical parameters:
   - Age
   - Gender
   - Total Bilirubin
   - Direct Bilirubin
   - Alkaline Phosphatase (ALP)
   - Alamine Aminotransferase (ALT/SGPT)
   - Aspartate Aminotransferase (AST/SGOT)
   - Total Proteins
   - Albumin
   - Albumin/Globulin Ratio

3. Click "Predict Liver Disease"
4. View the prediction result and probability score

## Input Parameters

| Parameter | Unit | Range | Description |
|-----------|------|-------|-------------|
| Age | years | 1-120 | Patient's age |
| Gender | - | Male/Female | Patient's gender |
| Total Bilirubin | mg/dL | 0.1-100.0 | Total bilirubin level |
| Direct Bilirubin | mg/dL | 0.0-50.0 | Direct bilirubin level |
| Alkaline Phosphatase | IU/L | 10-2000 | ALP enzyme level |
| ALT/SGPT | IU/L | 5-1000 | Alanine aminotransferase |
| AST/SGOT | IU/L | 5-1000 | Aspartate aminotransferase |
| Total Proteins | g/dL | 2.0-10.0 | Total protein level |
| Albumin | g/dL | 1.0-6.0 | Albumin level |
| A/G Ratio | - | 0.1-3.0 | Albumin to globulin ratio |

## Model Information

The application uses feature engineering to create additional predictive features:
- **AST/ALT Ratio**: Calculated from AST and ALT values
- **Albumin/Globulin Ratio**: Engineered from albumin and total protein values

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure all `.pkl` files are in the same directory as `streamlit_app.py`

2. **Import errors**
   - Check that all dependencies are installed: `pip install -r requirements.txt`

3. **Prediction errors**
   - Verify that input data matches the expected format
   - Check that feature names match those used during training

### Debug Mode

The application includes detailed error messages and debug information when predictions fail.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section above
- Create an issue in the repository
- Contact the development team

---

**Last Updated**: [Current Date]