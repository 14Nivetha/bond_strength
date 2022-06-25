from flask import Flask, render_template,request
import numpy as np
import pickle

#creating constructor
app=Flask(__name__)
model=pickle.load(open('model/model.pkl', 'rb'))
# print(model)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''v1 = request.form['cb']
    v2 = request.form['Mpa']
    v3 = request.form['Gpa']
    v4 = request.form['na']
    v5 = request.form['st']
    v6 = request.form['pr']'''


    features = [int(x) for x in request.form.values()]
    final_feature = [np.array(features)]
    pred = model.predict(final_feature)

    out = pred

    return render_template('index.html', prediction_text='Bond Strength of the concrete{}'.format(float(pred)))


if __name__ == '__main__':
    app.run(debug=True)
