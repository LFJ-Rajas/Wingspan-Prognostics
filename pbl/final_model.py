from flask import Flask, render_template, request, jsonify,redirect
import seaborn as sns
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from joblib import load
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(34)
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/redirect')
def redirect_page():
    return redirect('/pred')

@app.route('/pred')
def pr():
    return render_template('pred.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        unit_nr = int(request.form.get('unit_nr'))
        time_cycles = int(request.form.get('time_cycles'))
        setting_1 = float(request.form.get('setting_1'))
        setting_2 = float(request.form.get('setting_2'))
        setting_3 = float(request.form.get('setting_3'))
        s_1 = float(request.form.get('s_1'))
        s_2 = float(request.form.get('s_2'))
        s_3 = float(request.form.get('s_3'))
        s_4 = float(request.form.get('s_4'))
        s_5 = float(request.form.get('s_5'))
        s_6 = float(request.form.get('s_6'))
        s_7 = float(request.form.get('s_7'))
        s_8 = float(request.form.get('s_8'))
        s_9 = float(request.form.get('s_9'))
        s_10 = float(request.form.get('s_10'))
        s_11 = float(request.form.get('s_11'))
        s_12 = float(request.form.get('s_12'))
        s_13 = float(request.form.get('s_13'))
        s_14 = float(request.form.get('s_14'))
        s_15 = float(request.form.get('s_15'))
        s_16 = float(request.form.get('s_16'))
        s_17 = float(request.form.get('s_17'))
        s_18 = float(request.form.get('s_18'))
        s_19 = float(request.form.get('s_19'))
        s_20 = float(request.form.get('s_20'))
        s_21 = float(request.form.get('s_21'))

        arr = np.array([unit_nr,time_cycles,setting_1,setting_2, setting_3, s_1, s_2, s_3,s_4,s_5,s_6,s_7,s_8,s_9,s_10,s_11,s_12,s_13,s_14,s_15,s_16,s_17,s_18,s_19,s_20,s_21])
        inp_arr = arr.reshape(1,26)
        inp_arr_d = pd.DataFrame(inp_arr)
        columns_to_be_dropped = [0,1,2,3,4,5,9,10,14,20,22,23]
        inp_arr_first_col = inp_arr_d[0]
        scaler = MinMaxScaler(feature_range = (-1,1))
        inp_arr_d = scaler.fit_transform(inp_arr_d.drop(columns = columns_to_be_dropped))
        new_inp_arr= inp_arr_d.reshape(1,1,14)
        inp_arr = pd.DataFrame(data = np.c_[inp_arr_first_col, inp_arr_d])
        

        model = model = Sequential([
        layers.Conv1D(256, 3, activation = "relu", input_shape = (1,14),padding="same"),
        layers.Conv1D(96, 3, activation = "relu",padding="same"),
        layers.Conv1D(32, 3, activation = "relu",padding="same"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation = "relu"),
        layers.Dense(32, activation = "relu"),
        layers.Dense(1)
    ])
        model.compile(loss = "mse", optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001))
        rul_pred = model.predict(new_inp_arr)
        
        return render_template('pred.html',rul=int(abs(rul_pred[0][0]*10000)),unit_nr=unit_nr)
if __name__ == '__main__':
    app.run(debug=False)