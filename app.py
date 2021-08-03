from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Bar
import uuid
import os
import json
import plotly
import joblib

app = Flask(__name__)
attributes=["AGER_TYP" ,"CJT_GESAMTTYP" ,"D19_BANKEN_DIREKT" ,"D19_BANKEN_OFFLINE_DATUM " ,"D19_LEBENSMITTEL" ,"D19_LOTTO" ,"D19_SOZIALES", "D19_TECHNIK",
"D19_TIERARTIKEL" ,"FINANZ_HAUSBAUER" ,"GEBURTSJAHR","GFK_URLAUBERTYP" ,"GFK_URLAUBERTYP_3","KBA05_ANHANG ","KBA05_DIESEL" ,"KBA05_MAXHERST" 
,"KBA05_SEG2","KBA05_VORB0","KBA13_KRSHERST_FORD_OPEL" ,"KBA13_SEG_GELAENDEWAGEN" ,"KBA13_SEG_VAN" ,"PRAEGENDE_JUGENDJAHRE", "WOHNLAGE"]


train_null=pd.read_csv('data/train_missing.csv')
response=pd.read_csv('data/train_response.csv')

with open('data/model_ada_less.pkl', 'rb') as fp:
    ada = pickle.load(fp)

@app.route("/")
def hello_world():
    response_count = response.RESPONSE.value_counts().values
    response_names = [0,1]

    x_null=train_null.iloc[:,0].values
    y_null=train_null.iloc[:,1].values
    graphs = [
        {
            'data': [
                Bar(
                    x=response_names,
                    y=response_count
                )
            ],

            'layout': {
                'title': 'Customer Response Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Response"
                }
            }
        }
        ,{
            'data': [
                Bar(
                    x=x_null,
                    y=y_null
                )
            ],

            'layout': {
                'title': 'Trianset missing value percentage_top 20 variables',
                'yaxis': {
                    'title': "Missing percent"
                },
                'xaxis': {
                    'title': "varialbes"
                }
            }
        }
    ]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # save user input in query
    attribute_values=np.zeros((1,23))
    for i in range(len(attributes)):
        # print(attributes[i])
        # print(request.args.get(attributes[i], ''))
        attribute_values[0,i]=float(request.args.get(attributes[i], '') )
    # input_dict = dict(zip(attributes, attribute_values.tolist()[0]))

    # use model to predict classification for query
    prob=ada.predict_proba(attribute_values)[0][1]
    
    if prob>0.36:
        classification_label='A Potential Customer'
    else:
        classification_label='Not a Potential Customer'

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        classification_label=classification_label,attributes=attributes,values=attribute_values.tolist()[0]
    )
def main():
    app.run(host='127.0.0.1', port=3001, debug=True)

if __name__ == '__main__':
    main()
