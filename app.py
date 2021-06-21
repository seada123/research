#import relevant libraries for flask, html rendering and loading the ML model
from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import *

app = Flask(__name__)

#loading the SVM model and the preprocessor
model = pickle.load(open("modelrf.pkl", "rb"))
#std = pickle.load(open('std.pkl','rb'))
#model = load_model('lstm_model.h5')

#Index.html will be returned for the input
@app.route('/')
def hello_world():
    return render_template("index.html")


#predict function, POST method to take in inputs
@app.route('/predict',methods=['POST','GET'])
def predict():

    #take inputs for all the attributes through the HTML form
    USER_DEFINED_STATUS = request.form['1']
    FREQUENCY = request.form['2']
    RECORD_STAT = request.form['3']
    PRODUCT = request.form['4']
    PAYMENT_METHOD = request.form['5']
    TENOR = request.form['6']
    MAIN_COMP_RATE = request.form['7']
    RISK_FREE_EXP_AMOUNT = request.form['8']
    LCY_AMOUNT = request.form['9']
    

    #form a dataframe with the inpus and run the preprocessor as used in the training 
    row_df = pd.DataFrame([pd.Series([USER_DEFINED_STATUS, FREQUENCY, RECORD_STAT, PRODUCT, PAYMENT_METHOD, TENOR, MAIN_COMP_RATE, RISK_FREE_EXP_AMOUNT,LCY_AMOUNT])])
    
#   row_df =  pd.DataFrame(std.transform(row_df))



    #scaler = StandardScaler()
    #row_df = scaler.fit_transform(row_df)
    #row_df = row_df.reshape((1, row_df.shape[1]))
    #print(row_df)
    
   # row_df= pd.to_numeric(row_df)
    
    prediction=model.predict(row_df)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = prediction
    #output_print = str(float(output)*100)+'%'
    #output_print = str(float(output)
    

    if float(output)<0.5:
         return render_template('result.html',pred=f'The Credit Applicant Is Eligible to Get the Requested Credit')
         #return render_template('result.html',pred=output)
    else:
        return render_template('result.html',pred=f'The Credit Applicant Is Not Eligible to Get the Requested Credit')
        #return render_template('result.html',pred=output)
    #return render_template("predictor.html")


if __name__ == '__main__':
    app.run(debug=True)
