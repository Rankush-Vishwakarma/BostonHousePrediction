from flask import *
import os
#from deployment import *
from werkzeug.utils import secure_filename
app = Flask(__name__)
import pickle
#from loggingModule import makeLog
import numpy as np
#model , tokenizer = load_Model_Tokenizer()
#log = makeLog()
import sys
'''
logging works in desktop when the server is in locally running. 
'''

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
def loadModel():
    try:
        with open(r"finalized_model.sav", "rb") as LR_input:
            LR = pickle.load(LR_input)
        with open(r"lasso_model.sav", "rb") as LR_input:
            LassoModel = pickle.load(LR_input)
        with open(r"ridge_model.sav", "rb") as LR_input:
            RidgeModel = pickle.load(LR_input)
        #log.info('All the models have been loaded sucessfully.')
        #print('All the models have been loaded sucessfully.')
        return LR, LassoModel, RidgeModel
    except Exception as e:
        #log.error(e)
        #print(str(e))
        pass
def predict(data):
    try:
        model = {}
        data = [float(i) for i in data]
        d = np.array(data).reshape(1,-1)
        LR, LS, RM = loadModel()
        model['Linear Regression Prediction'] = LR.predict(d)[0]
        model['Lasso Regression Prediction'] = LS.predict(d)[0]
        model['Ridge Regression Prediction'] = RM.predict(d)[0]
        model['Average Prediction Value'] = ((LR.predict(d)[0]+LS.predict(d)[0]+RM.predict(d)[0])/3)
        #log.info('Model Predicted successfully.')
        #print('Model Predicted successfully.')
        return model
    except Exception as e:
        #log.error(e)
        #print(e)
        pass
@app.route('/')  
def home():
    try:
        return render_template("home.html")  
    except Exception as e:
        #log.error(e)
        #print(e)
        pass
 
@app.route('/', methods = ['POST','GET'])  
def success():
    try:
        try:
            if request.method == 'GET':
                return make_response('failure')
            #log.debug('HTTP response is not right, please debug it.')
            #print('HTTP response is not right, please debug it.')
        except Exception as e:
            #log.error(e)
            #print(e)
            pass
        try:
            if request.method == 'POST':
                try:
                    result = request.form
                    data = result.to_dict(flat=True).values()
                    #print(data)
                    #log.info('Data has been successfully populated')
                    #print('Data has been successfully populated')
                except Exception as e:
                    #log.debug('Data is not getting populated.')
                    #print('Data is not getting populated.')
                    pass
                try:
                    a = predict(data)
                except Exception as e:
                    #log.error(e)
                    #print(e)
                    pass
                def dict2htmltable(data):
                    try:
                        #html = '<thead>' + 'boston data prediction' + '</thead>'
                        html = ''.join('<th>' + str(x) + '</th>' for x in data[0].keys())
                        for d in data:
                            html += '<tr style=" border: 1px solid white;">' + ''.join('<td style="text-align:center; border: 1px solid white;">' + str(round(x,2)) + '</td>' for x in d.values()) + '</tr>'
                        final_html = """<!DOCTYPE html><html lang="en" xmlns="http://www.w3.org/1999/xhtml">
                            <head>
                            <meta charset="utf-8" />
                                <title>Result</title>
                            </head>
                            <body style="background-image: radial-gradient(circle, #353535, #313439, #27343c, #19353b, #0a3535); padding:50px">
                                <div class="container" style="color:white"  font-size:20px; ">
                                    <h1  "><b>Prediction</b></h1>
                                </div>
                                <div class="container" style="width: 800px; margin-top:5%; margin-left:20%; color:white;">
                                    {0}
                                </div>
            
                            </body>
                            </html>"""
                        #log.info('Prediction data has been successfully populated.')
                        #print('Prediction data has been successfully populated.')
                    except Exception as e:
                        #log.error(e)
                        #print(e)
                        pass
                    return final_html.format('<table style = "border: 3px solid white;margin-left:auto;margin-right:auto; font-size:20px color:white; " id="table1">' + html + '</table>')
                html = dict2htmltable([a])
                with open("templates/table.html", "w") as file:
                    file.write(html)    
            return render_template("table.html")
        except Exception as e:
            #log.error(e)
            #print(e)
            pass
    except Exception as e:
        #log.error(e)
        #print(e)
        pass
if __name__ == '__main__':  
    app.run()  
