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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
  #  Occupation={'0':'Entrepreneur','1':'Self_Employed','2':'Other','3':'Salaried'}
  #  Channel_Code={'0':'X1','1':'X2','2':'X3','3':'X4'}
  #  Credit_Product={'0':'YES','1':'NO','2':'NA'}
  #  Is_Active={'0':'YES','1':'NO'}
    
    final_features=[]
    input=request.form.to_dict()
    value = list(input.values())
   # if(value[0]=='0'):
    #    value[0]='Male'
   # else:
     #   value[0]='Female'
   # value[2]='RG'+str(250+int(value[2]))
#    value[3]=Occupation[value[3]]
#    value[4]=Channel_Code[value[4]]
#    value[6]=Credit_Product[value[6]]
#    value[8]=Is_Active[value[8]]
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

    return render_template("home.html", prediction = output)




        
        
        
        
        
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
