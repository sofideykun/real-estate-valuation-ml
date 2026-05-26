import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import joblib

# Загрузка данных
df = pd.read_csv("data/saint_petersburg_housing_data.csv")
print("Размер датасета:", df.shape)

# Визуализация зависимостей (EDA)
print("Генерация графиков зависимостей...")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="total_area", y=df["price"] / 1_000_000, alpha=0.35)
plt.title("Цена vs площадь")
plt.xlabel("Общая площадь (м²)")
plt.ylabel("Цена (млн руб.)")
plt.tight_layout()
plt.savefig("price_vs_area.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="floor", y=df["price"] / 1_000_000, alpha=0.35, color="green")
plt.title("Цена vs этаж")
plt.xlabel("Этаж")
plt.ylabel("Цена (млн руб.)")
plt.tight_layout()
plt.savefig("price_vs_floor.png")
plt.close()

# Числовые признаки
num_features = [
    "floor",
    "total_area",
    "living_area",
    "kitchen_area",
    "year",
    "flat_status",
    "metro_minutes",
]
# Масштабируем числовые признаки
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(df[num_features].values)

# Текстовые признаки через TF-IDF
# Добавление стоп-слов 
stop_words_ru = ["и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой", "им", "более", "всегда", "конечно", "всю", "между"]
tfidf = TfidfVectorizer(max_features=100, stop_words=stop_words_ru, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df["description"])

print("\nСлова из TF-IDF словаря:")
print(tfidf.get_feature_names_out())

# Объединение числовых и текстовых признаков
X = hstack([X_num_scaled, X_text])
y = df["price"].values

print("\nРазмер матрицы признаков:", X.shape)
print("Из них числовых:", len(num_features))
print("Из них текстовых (TF-IDF):", 100)

# Разделение на обучение и тест
X_num_train, X_num_test, y_train, y_test = train_test_split(X_num_scaled, y, test_size=0.2, random_state=42)
X_combined_train, X_combined_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

datasets = {
    "Только числа": (X_num_train, X_num_test),
    "Числа + Текст": (X_combined_train, X_combined_test)
}

# Три модели
base_models = {
    "Linear Regression": LinearRegression(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}

for name, base_model in base_models.items():
    for data_name, (X_train, X_test) in datasets.items():
        model_name = f"{name} ({data_name})"
        print(f"\nОбучаем {model_name}...")
        
        model = clone(base_model)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        results[model_name] = {"model": model, "r2": r2, "mae": mae, "mape": mape}

        print(f"  R²:   {r2:.4f}")
        print(f"  MAE:  {mae:,.0f} руб.")
        print(f"  MAPE: {mape:.2%}")

        # График факт vs предсказание
        plt.figure(figsize=(8, 6))
        y_test_mln = y_test / 1_000_000
        y_pred_mln = y_pred / 1_000_000
        plt.scatter(y_test_mln, y_pred_mln, alpha=0.4)
        plt.xlabel("Фактическая цена (руб.)")
        plt.ylabel("Предсказанная цена (руб.)")
        plt.title(f"{model_name}: факт vs прогноз")
        
        # 1. Изменили линию идеального прогноза, чтобы она шла от 0 до 100
        plt.plot([0, 100], [0, 100], "r--")
        
        # 2. Жестко ограничиваем оси графика от 0 до 100 млн
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        mln_rub_formatter = FuncFormatter(lambda x, pos: f"{x:.0f} млн руб.")
        plt.gca().xaxis.set_major_formatter(mln_rub_formatter)
        plt.gca().yaxis.set_major_formatter(mln_rub_formatter)
        
        # (Опционально) Для шкалы 0-100 лучше смотрятся шаги по 20, а не 25
        plt.gca().xaxis.set_major_locator(MultipleLocator(20))
        plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
        plt.gca().yaxis.set_major_locator(MultipleLocator(20))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(10))
        plt.xticks(rotation=25, ha="right")
        plt.tick_params(axis="both", which="minor", length=3, color="gray")
        plt.grid(which="major", alpha=0.25)
        plt.grid(which="minor", alpha=0.12)
        plt.tight_layout()
        plt.savefig(f"{model_name.lower().replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '')}_pred_vs_real.png")
        plt.close()

# Сохранение метрик в CSV
metrics_df = pd.DataFrame(
    {name: {"R2": v["r2"], "MAE": v["mae"], "MAPE": v["mape"]} for name, v in results.items()}
).T
metrics_df.to_csv("model_metrics.csv")
print("\nМетрики сохранены в model_metrics.csv")

# График сравнения моделей по R²
plt.figure(figsize=(6, 4))
sns.barplot(
    x=metrics_df.index,
    y=metrics_df["R2"],
    color=sns.color_palette("viridis", 3)[1],
)
plt.ylabel("R²")
plt.title("Сравнение моделей по R²")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_r2_comparison.png")
plt.close()

# Важность числовых признаков у Random Forest
rf_model = results["Random Forest (Числа + Текст)"]["model"]
num_importances = rf_model.feature_importances_[: len(num_features)]

plt.figure(figsize=(6, 4))
sns.barplot(
    x=num_importances,
    y=num_features,
    hue=num_features,
    palette="Blues_r",
    legend=False,
)
plt.title("Важность числовых признаков (Random Forest)")
plt.xlabel("Важность признака (доля вклада в модель)")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.close()

# Важность текстовых признаков у Random Forest
text_importances = rf_model.feature_importances_[len(num_features) :]
tfidf_words = tfidf.get_feature_names_out()

top_n = 15
top_idx = text_importances.argsort()[-top_n:][::-1]
top_words = [tfidf_words[i] for i in top_idx]
top_vals = [text_importances[i] for i in top_idx]

plt.figure(figsize=(7, 5))
sns.barplot(
    x=top_vals,
    y=top_words,
    hue=top_words,
    palette="Oranges_r",
    legend=False,
)
plt.title("Топ-15 слов из описания по важности (Random Forest)")
plt.xlabel("Важность признака (доля вклада в модель)")
plt.tight_layout()
plt.savefig("tfidf_importance.png")
plt.close()

print("Графики сохранены")

# Выбираем лучшую модель среди тех, что обучались на тексте (Числа + Текст)
text_models = [name for name in results.keys() if "(Числа + Текст)" in name]
best_name = max(text_models, key=lambda x: results[x]["r2"])
best_model = results[best_name]["model"]

joblib.dump(best_model, "model.joblib")
joblib.dump(tfidf, "tfidf.joblib")
joblib.dump(scaler, "scaler.joblib")

print(f"\nВыбрана лучшая модель с текстом: {best_name} (R² = {results[best_name]['r2']:.4f})")
print("Модель сохранена в model.joblib")
print("Скейлер сохранён в scaler.joblib")
print("TF-IDF сохранён в tfidf.joblib")