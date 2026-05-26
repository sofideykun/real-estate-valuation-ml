import psycopg2
import pandas as pd

conn = psycopg2.connect(
    database="Real_estate",      
    user="postgres",              
    password="postgres",          
    host="127.0.0.1",             
    port="5432"                   
)
conn.autocommit = True
cursor = conn.cursor()

df = pd.read_csv("data/saint_petersburg_housing_data.csv")

df['description'] = df['description'].fillna('')

for i, row in df.iterrows():
    floor = int(row['floor'])
    total_area = float(row['total_area'])
    living_area = float(row['living_area'])
    kitchen_area = float(row['kitchen_area'])
    year = int(row['year'])
    flat_status = int(row['flat_status'])
    metro_minutes = int(row['metro_minutes'])
    description = str(row['description'])
    price = float(row['price'])

    cursor.execute("""
        INSERT INTO real_estate_objects 
        (floor, total_area, living_area, kitchen_area, year, flat_status, metro_minutes, description)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        floor, total_area, living_area, kitchen_area, 
        year, flat_status, metro_minutes, description
    ))
    
    object_id = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO prices (object_id, price_value)
        VALUES (%s, %s)
    """, (
        object_id, price
    ))

    cursor.execute("SELECT * FROM real_estate_objects LIMIT 100")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

print("Данные из CSV успешно загружены в PostgreSQL!")

cursor.close()
conn.close()
