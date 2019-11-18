import flask
import dill 
import numpy as np
import pandas as pd 

app = flask.Flask(__name__)

with open('RandomForest_titanic.pkl', 'rb') as f:
    PREDICTOR = dill.load(f)
##################################
@app.route("/")
def hello():
    return '''
    <body>
    <h2> Hello World! <h2>
    </body>
    '''

##################################
@app.route('/greet/<name>')
def greet(name):
    '''Say hello to your first parameter'''
    return "Hello, %s!" %name

@app.route('/predict', methods=["GET"])
def predict():
    pclass = flask.request.args['pclass']
    sex = flask.request.args['sex']
    age = flask.request.args['age']
    fare = flask.request.args['fare']
    sibsp = flask.request.args['sibsp']

    item = pd.DataFrame([[pclass, sex, age, fare, sibsp]], columns=['pclass', 'sex', 'age', 'fare', 'sibsp'])
    
    #item = np.array([pclass, sex, age, fare, sibsp])
    print (item)
    score = PREDICTOR.predict_proba(item)
    results = {'survival chances': score[0,1], 'death chances': score[0,0]}
    return flask.jsonify(results)

##################################
#@app.route('/page')
#def show_page():
#    return flask.render_template('dataentrypage.html')

##################################
@app.route('/page', methods=['POST', 'GET'])
def page():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

       inputs = flask.request.form

       pclass = inputs['pclass'][0]
       sex = inputs['sex'][0]
       age = inputs['age'][0]
       fare = inputs['fare'][0]
       sibsp = inputs['sibsp'][0]

       item = pd.DataFrame([[pclass, sex, age, fare, sibsp]], columns=['pclass', 'sex', 'age', 'fare', 'sibsp'])
       print (item)
       score = PREDICTOR.predict_proba(item)
       #results = {'survival chances': score[0,1], 'death chances': score[0,0]}
       survive = int(score[0,1] * 100)
       dead = int(score[0,0] * 100)
    else:
        survive = 0
        dead = 0
    return flask.render_template('dataentrypage.html', survive=survive, dead=dead)

##################################
if __name__ == '__main__':
    app.run(debug=True)