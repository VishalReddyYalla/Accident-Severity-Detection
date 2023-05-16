
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier



webapp=Flask(__name__)

@webapp.route('/')
def index():
    return render_template('index.html')
@webapp.route('/load',methods=["GET","POST"])
def load():
    global df,dataset
    if request.method=="POST":
        file=request.files['file']
        df=pd.read_excel(file,engine='openpyxl')
        dataset=df.head(100)
        msg='Data Loaded Successfully'
        return render_template('load.html',msg=msg)
    return render_template('load.html')
@webapp.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@webapp.route('/preprocess',methods=['POST','GET'])
def preprocess():

    global X,y,X_train, X_test, y_train, y_test,df,X
    if request.method=="POST":
        size=int(request.form['split'])
        size=size/100
        df.dropna(axis=0, inplace=True)
        df.drop_duplicates(inplace=True)
        le = LabelEncoder()
        df['Road Surface'] = le.fit_transform(df['Road Surface'].astype(str))
        df['Lighting Conditions'] = le.fit_transform(df['Lighting Conditions'])
        df['Weather Conditions'] = le.fit_transform(df['Weather Conditions'])
        df['Casualty Class'] = le.fit_transform(df['Casualty Class'])
        df['Sex of Casualty'] = le.fit_transform(df['Sex of Casualty'])
        df['Type of Vehicle'] = le.fit_transform(df['Type of Vehicle'])
        df['Reference Number'] = le.fit_transform(df['Reference Number'])
        df['Easting'] = le.fit_transform(df['Easting'])
        df['Northing'] = le.fit_transform(df['Northing'])
        df['Number of Vehicles'] = le.fit_transform(df['Number of Vehicles'])
        df['Accident Date'] = le.fit_transform(df['Accident Date'])
        df['Time (24hr)'] = le.fit_transform(df['Time (24hr)'])
        df['1st Road Class'] = le.fit_transform(df['1st Road Class'])
        df['Age of Casuality'] = le.fit_transform(df['Age of Casuality'])
        df['Casualty Severity'] = le.fit_transform(df['Casualty Severity'])
        print(df.head())
        df.to_csv('cleaned.csv')
        from imblearn.over_sampling import (SMOTE, RandomOverSampler)
        oversampled = SMOTE()
        X = df.iloc[:,:-1]
        y = df.iloc[:,-1]

        X = pd.DataFrame(X, columns=['Number of Vehicles', 'Time (24hr)', '1st Road Class','Road Surface',
       'Lighting Conditions', 'Weather Conditions', 'Casualty Class',
       'Sex of Casualty', 'Age of Casuality', 'Speed', 'Type of Vehicle'])
        y = pd.DataFrame(y, columns=['Casualty Severity'])
        X, y = oversampled.fit_resample(X, y)

        print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

        return render_template('preprocess.html',msg='Data Preprocessed and It Splits Successfully')
    msg=""
    return render_template('preprocess.html',msg=msg)

@webapp.route('/model',methods=['POST','GET'])
def model():
    try:
        if request.method=="POST":
            print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
            s=int(request.form['algo'])
            if s==0:
                return render_template('model.html',msg='Please Choose an Algorithm to Train')
            elif s==1:
                print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
                from sklearn.metrics import accuracy_score
                rf = RandomForestClassifier(n_estimators=300, random_state=35).fit(X_train, y_train)
                print('aaaaaaaaaaaaaaaaaaaaaaaaa')
                rf_pred = rf.predict(X_test)
                rf_dt = accuracy_score(y_test, rf_pred)
                rf_dt = rf_dt * 100
                msg = 'The accuracy obtained by RandomForestClassifier is ' + str(rf_dt) + str('%')

                return render_template('model.html', msg=msg)
            elif s==2:
                from sklearn.metrics import accuracy_score
                xgb = XGBClassifier(learning_rate=0.07, n_estimators=300,
                              class_weight="balanced_subsample",
                              max_depth=8, min_child_weight=1,
                              scale_pos_weight=7,
                              seed=27, subsample=0.8, colsample_bytree=0.8).fit(X_train,y_train)

                xgb_pred=xgb.predict(X_test)
                xgb_dt = accuracy_score(y_test, xgb_pred)
                xgb_dt=xgb_dt*100
                msg = 'The accuracy obtained by XGBClassifier is ' + str(xgb_dt) + str('%')
                return render_template('model.html', msg=msg)
            elif s==3:
                from sklearn.metrics import accuracy_score
                from sklearn.svm import SVC
                sv = SVC()
                sv.fit(X_train,y_train)
                sv_pred=sv.predict(X_test)
                sv_dt = accuracy_score(y_test, sv_pred)
                sv_dt=sv_dt*100
                msg = 'The accuracy obtained by svc is ' + str(sv_dt) + str('%')
                return render_template('model.html', msg=msg)
            elif s==4:
                from sklearn.metrics import accuracy_score
                lg = LogisticRegression(solver='saga', max_iter=1000).fit(X_train, y_train)
                pred = lg.predict(X_test)
                ac_lg = accuracy_score(y_test, pred)
                ac_lg = ac_lg * 100
                msg = 'The accuracy obtained by Logistic Regression ' + str(ac_lg) + str('%')
                return render_template('model.html', msg=msg)
            elif s==5:
                from sklearn.metrics import accuracy_score
                params = {'learning_rate': 0.1, 'depth': 6, \
                          'l2_leaf_reg': 3, 'iterations': 100}

                cb = CatBoostClassifier(**params)
                cb.fit(X_train,y_train)
                cb_pred =cb.predict(X_test)
                cb_dt = accuracy_score(y_test, cb_pred)
                cb_dt=cb_dt*100
                msg = 'The accuracy obtained by catboost is ' + str(cb_dt) + str('%')
                return render_template('model.html', msg=msg)
            return render_template('model.html')
        return render_template('model.html')
    except:
        msg='Please Upload the required dataset'
        return render_template('load.html',msg=msg)

@webapp.route('/prediction',methods=["GET","POST"])
def prediction():
        if request.method=="POST":
            f1 = int(request.form['LightingConditions'])
            f2 = int(request.form['RoadSurface'])
            f3 = int(request.form['WeatherConditions'])
            f4 = int(request.form['CasualtyClass'])
            f5 = int(request.form['SexofCasualty'])
            f6 = int(request.form['TypeofVehicle'])
            f7 = int(request.form['speed'])
            f8 = int(request.form['NoofVehicle'])
            f9 = int(request.form['Age of Casuality'])
            f10= int(request.form['Time (24hr)'])
            f11= int(request.form['1st Road Class'])
            le = LabelEncoder()
            li=le.fit_transform([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11])

            print(li)
            print("dfsfkhkfhksfhksfh")
            model=RandomForestClassifier()
            model.fit(X_train,y_train)
            result=model.predict([li])
            print(result)

            if result == 0:
                msg = 'Fatal'
                return render_template('prediction.html', msg=msg)
            elif result ==1:
                msg = 'Serious'
                return render_template('prediction.html', msg=msg)
            elif result ==2:
                msg = 'Slight'
                return render_template('prediction.html', msg=msg)
        return render_template('prediction.html')


if __name__=='__main__':
    webapp.run(debug=False,host='0.0.0.0')
#,host='0.0.0.0'
  