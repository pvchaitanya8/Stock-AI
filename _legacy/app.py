# app.py

from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib
matplotlib.use('agg')  # Use the "agg" backend for Matplotlib
import matplotlib.pyplot as plt


app = Flask(__name__)

# Load the trained model
model = None

@app.before_first_request
def load_model():
    global model
    # Load the model architecture from JSON file
    json_file = open('modelGG.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load the model weights
    model.load_weights('modelGG.h5')

@app.route('/')

@app.route('/predict', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        import matplotlib.pyplot as plt
        company = request.form['company']
        opening = float(request.form['opening'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        volume = float(request.form['volume'])
         
        ind = company

        if(company=="Apple"):
            company=0
            df1= pd.read_csv(r'Data\0.csv')
        elif(company=="Cisco"):
            company=1
            df1= pd.read_csv(r'Data\1.csv')
        elif(company=="CocaCola"):
            company=2
            df1= pd.read_csv(r'Data\2.csv')
        elif(company=="Google"):
            company=3
            df1= pd.read_csv(r'Data\3.csv')
        elif(company=="IBM"):
            company=4
            df1= pd.read_csv(r'Data\4.csv')
        elif(company=="Intel"):
            company=5
            df1= pd.read_csv(r'Data\5.csv')
        elif(company=="JPMorgan"):
            company=6
            df1= pd.read_csv(r'Data\6.csv')
        elif(company=="Microsoft"):
            company=7
            df1= pd.read_csv(r'Data\7.csv')
        elif(company=="McDonalds"):
            company=8
            df1= pd.read_csv(r'Data\8.csv')
        elif(company=="Nike"):
            company=9
            df1= pd.read_csv(r'Data\9.csv')
        elif(company=="Verizon"):
            company=10
            df1= pd.read_csv(r'Data\10.csv')
        elif(company=="Visa"):
            company=11
            df1= pd.read_csv(r'Data\11.csv')
        elif(company=="Walmart"):
            company=12
            df1= pd.read_csv(r'Data\12.csv')
        elif(company=="Disney"):
            company=13
            df1= pd.read_csv(r'Data\13.csv')



        # Preprocess the input data
        val = [opening, high, low, volume]
        input_data = np.array([[company, opening, high, low, volume]])
        
        # Make predictions using the loaded model
        prediction = model.predict(input_data)

        
        #__ Ploting Graphs
        orginal = df1.loc[:,"Adj Close"]
        orginal = orginal.values.tolist()
        if prediction < orginal[-1]:
            flag=0
        else:
            flag=1  
            
        orginal.append(prediction)

        if flag==0:
            with plt.style.context('dark_background'):
                plt.plot(orginal, color = 'red')
        else:
            with plt.style.context('dark_background'):
                plt.plot(orginal, color = 'green')
        #__ saving the graph
        plt.savefig('static/plot.png')
        
        # Format the prediction result
        result = {'company': ind, 'prediction': float(prediction[0][0]), 'inputs': val}

        return render_template("index.html", output_data = result)
    return render_template("index.html", output_data = "")
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
