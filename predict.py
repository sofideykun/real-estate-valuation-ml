import joblib
from scipy.sparse import hstack

MODEL_PATH = 'model.joblib'
SCALER_PATH = 'scaler.joblib'
TFIDF_PATH = 'tfidf.joblib'

def predict_price(floor, total_area, living_area, kitchen_area,
                  year, flat_status, metro_minutes, description):

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    # Числовые признаки
    num_features = [[floor, total_area, living_area,
                     kitchen_area, year, flat_status, metro_minutes]]
                     
    # Масштабируем признаки так же, как при обучении
    X_num_scaled = scaler.transform(num_features)
    
    # Текстовые признаки из описания
    X_text = tfidf.transform([description])

    # Объединяем масштабированные числа и текст
    X = hstack([X_num_scaled, X_text])

    # Предсказываем
    prediction = model.predict(X)
    return float(prediction[0])