def prepare_features(form):
    try:
        floor         = int(form['floor'])
        total_area    = float(form['total_area'])
        living_area   = float(form['living_area'])
        kitchen_area  = float(form['kitchen_area'])
        year          = int(form['year'])
        flat_status   = int(form['flat_status'])
        metro_minutes = int(form['metro_minutes'])
        description   = form['description'].strip()

        return floor, total_area, living_area, kitchen_area, year, flat_status, metro_minutes, description

    except (KeyError, ValueError):
        raise ValueError("Ошибка: проверьте правильность введённых данных.")