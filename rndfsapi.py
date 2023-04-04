#Install Libraries
from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import numpy as np
import sys
import pyshark

#LOCAL_IP = '192.168.0.103'
#LOCAL_IPV6 = 'fe80::a580:6360:f23b:33c3%10'
LOCAL_IP = '192.168.168.202'
LOCAL_IPV6 = '44:2f:bd:7d:d9'
REMOTE_PORTS = [3341, 3333, 3334, 3357, 80, 443]
feature_col = ["int_time", "pkt_size", "mm_it", "mm_ps", "mstd_it", "mstd_ps"]

application = Flask(__name__)


@application.route("/prediction", methods=["POST"])


#define function
def predict():
    if lr:
        try:
            #json_ = request.json
            json_ = data_man()
            #print(json_)
            #query = pd.get_dummies(pd.DataFrame(json_))
            query = pd.read_json(json_)
            query = query.reindex(columns=rnd_columns)

            predict = list(lr.predict(query))
            return jsonify({"prediction": str(list(i for i in predict))})
        except:
            return jsonify({"trace": traceback.format_exc()})
    else:
        print ("Model not good")
        return "Model is not good"




def data_man():
    df = pd.DataFrame(t, columns=['int_time', 'pkt_size'])
    df['mm_it'] = df['int_time'].rolling(window=5).mean()
    df['mm_ps'] = df['pkt_size'].rolling(window=5).mean()
    df['mstd_it'] = df['int_time'].rolling(window=5).std()
    df['mstd_ps'] = df['pkt_size'].rolling(window=5).std()
    df.drop(df.index[0:4], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df[feature_col]

    scaler = preprocessing.RobustScaler()

    X_arr = scaler.fit_transform(df)
    X = pd.DataFrame(X_arr, columns=feature_col)

    stnd = preprocessing.StandardScaler()

    X_arr = stnd.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=feature_col)
    return X.to_json(orient="records")
    

if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except:
        port = 12345
    #capt = pyshark.LiveCapture(interface='Wi-Fi')
    #capt = pyshark.FileCapture('fludg8.000webhostapp_pcap.pcap')
    t = [[0.10321593284606934, 52], [0.09416317939758301, 40], [0.005105018615722656, 4360], [0.0, 499],
        [0.0948488712310791, 314], [0.0, 607], [0.007678985595703125, 52], [0.21971392631530762, 40],
        [0.00067901611328125, 2920], [0.0006148815155029297, 1256], [0.0, 52], [0.0, 1480],
        [0.0001709461212158203, 664], [0.30251288414001465, 91], [0.0, 40]]
    #lr = pickle.load(open("xgb_x_rs_3_24_booster_bin.bin", "rb"))
    lr = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model('xgb_x_rs_3_24_booster_bin.bin')
    lr._Booster = booster
    #lr._le = preprocessing.LabelEncoder().fit([0, 1])
    print ("Model loaded")
    rnd_columns = pickle.load(open("rnd_columns.pkl", "rb")) # Load “rnd_columns.pkl”
    print ("Model columns loaded")
    application.run(port=port, debug=True)