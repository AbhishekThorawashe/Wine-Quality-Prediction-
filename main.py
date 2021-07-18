
# importing the necessary dependencies
from flask import Flask, render_template, request,jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            fixed_acidity =float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            #filename_scaler = 'WineStandardScaler.pickle'
            #filename = 'wine_model.pickle'

            # loading the model file from the storage
            with open("WineStandardScaler.sav", 'rb') as f:
                scalar = pickle.load(f)
            with open("EnsembleModelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)

            # predictions using the loaded model file
            scaled_data= scalar.transform([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                                                 total_sulfur_dioxide, density, pH, sulphates, alcohol]])
            prediction=model.predict(scaled_data)
            print('prediction is', prediction[0])
            if prediction[0]==3:
                result='Bad'
            elif prediction[0]==4:
                result='Below Average'
            elif prediction[0]==5:
                result='Average'
            elif prediction[0]==6:
                result='Good'
            elif prediction[0]==7:
                result='Very Good'
            else :
                result='Excellent'


            # showing the prediction results in a UI
            return render_template('results.html',prediction=result)
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5002, debug=True)
	#app.run(debug=True) # running the app