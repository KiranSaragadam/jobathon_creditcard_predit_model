import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
genderclf = pickle.load(open('Gender', 'rb'))
Occupationclf = pickle.load(open('Occupation', 'rb'))
Region_Codeclf = pickle.load(open('Region_Code', 'rb'))
Is_Activeclf = pickle.load(open('Is_Active', 'rb'))
Credit_Productclf = pickle.load(open('Credit_Product', 'rb'))
Channel_Codeclf = pickle.load(open('Channel_Code', 'rb'))
model = pickle.load(open('model', 'rb'))
feature_importance = pickle.load(open('feature_importance', 'rb'))
feature = pickle.load(open('feature', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
   
    final_features=[]
    input=request.form.to_dict()
    value = list(input.values())
    final_features.extend(list(genderclf.transform([value[0]]).toarray()[0]))
    final_features.append(int(value[1]))
    final_features.extend(list(Region_Codeclf.transform([value[2]]).toarray()[0]))
    final_features.extend(list(Occupationclf.transform([value[3]]).toarray()[0]))
    final_features.extend(list(Channel_Codeclf.transform([value[4]]).toarray()[0]))
    final_features.append(int(value[5]))
    final_features.extend(list(Credit_Productclf.transform(np.array(['No']).reshape(1,1)).toarray()[0]))
    final_features.append(int(value[7]))
    final_features.extend(list(Is_Activeclf.transform([value[8]]).toarray()[0]))
    final_features = np.array(final_features).reshape(1, 52)
    prediction = model.predict(final_features)

    if (prediction[0]==1):
        output='Yes'
    else:
        output='No'
    proba=model.predict_proba(final_features)
    class1=str(proba[0][max(prediction[0],1)])
    class0=str(proba[0][min(prediction[0],0)])
    imp=np.array(feature)[np.where(final_features[0]>=1)][np.argsort(feature_importance[np.where(final_features[0]>=1)])[::-1]]
    return render_template("home.html", prediction = output,prediction_1=str(class0),prediction_2=str(class1),prediction_3=str(','.join(imp)))




        
        
        
        
        
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
