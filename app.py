from flask import Flask, render_template, request
from predict import predict_price
from database import RealEstateObject, Price, Prediction, SessionLocal, init_db
from signs import prepare_features

app = Flask("app")

init_db()

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        floor, total_area, living_area, kitchen_area, \
        year, flat_status, metro_minutes, description = prepare_features(request.form)

        predicted = predict_price(floor, total_area, living_area, kitchen_area,
                                  year, flat_status, metro_minutes, description)

        db = SessionLocal()

        # 1. Создаем Объект недвижимости
        real_estate = RealEstateObject(
            floor=floor,
            total_area=total_area,
            living_area=living_area,
            kitchen_area=kitchen_area,
            year=year,
            flat_status=flat_status,
            metro_minutes=metro_minutes,
            description=description
        )
        db.add(real_estate)
        db.flush() # Получаем ID созданного объекта

        # 2. Создаем сущность Цена
        price_record = Price(
            object_id=real_estate.id,
            price_value=None
        )
        db.add(price_record)

        # 3. Создаем сущность Предсказание
        prediction_record = Prediction(
            object_id=real_estate.id,
            model_name="Random Forest (Числа + Текст)",
            predicted_price=round(predicted, 2)
        )
        db.add(prediction_record)

        db.commit()
        db.close()

        return render_template('result.html', price=f"{predicted:_.0f}".replace('_', ' '))

    except Exception as e:
        return f"Ошибка: {e}"