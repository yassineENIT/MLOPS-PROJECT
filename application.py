import joblib
import numpy as np
from config.paths_config import MODEL_OUTPUT_PATH
from flask import Flask, request, render_template



app=Flask(__name__)
loaded_model=joblib.load(MODEL_OUTPUT_PATH)
@app.route('/',methods=['GET','POST'])

def index():
    if request.method=='POST':
        lead_time= int(request.form['lead_time'])
        no_of_special_requests=int(request.form['no_of_special_requests'])
        avg_price_per_room=float(request.form['avg_price_per_room'])
        arrival_month=int(request.form['arrival_month'])
        arrival_date=int(request.form['arrival_date'])

        market_segment_type=int(request.form['market_segment_type'])
        no_of_weekend_nights=int(request.form['no_of_weekend_nights'])
        no_of_week_nights=int(request.form['no_of_week_nights'])
        type_of_meal_plan=int(request.form['type_of_meal_plan'])
        room_type_reserved=int(request.form['room_type_reserved'])

        features=np.array([[lead_time,no_of_special_requests,avg_price_per_room,arrival_month,arrival_date,market_segment_type,no_of_weekend_nights,no_of_week_nights,type_of_meal_plan,room_type_reserved]])
        prediction=loaded_model.predict(features)
        return render_template('index.html',prediction_text=f'The predicted hotel booking is : {prediction[0]}')
    return render_template('index.html',prediction=None)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080,debug=True) 





