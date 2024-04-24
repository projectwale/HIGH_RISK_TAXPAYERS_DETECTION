from flask import render_template, request, redirect, url_for, session, Flask, flash
import re
import pickle
from pickle import dump, load
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import string
import random
import pymysql

app=Flask(__name__)
app.secret_key = 'super secret key'

def dbConnection():
    try:
        connection = pymysql.connect(host="localhost", user="root", password="root", database="stockmarket")
        return connection
    except:
        print("Something went wrong in database Connection")

def dbClose():
    try:
        dbConnection().close()
    except:
        print("Something went wrong in Close DB Connection")

con=dbConnection()
cursor=con.cursor()

ALLOWED_EXTENSIONS = {'xlsx', 'xlsm', 'csv'}
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        result_count = cursor.execute('SELECT * FROM tblregister WHERE email = %s AND password = %s', (email, password))
        result = cursor.fetchone()
        dbClose()
        print("result")
        print(result)
        if result_count>0:
            session['loggedin'] = True
            session['userid'] = result[0]
            session['name'] = result[1]
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect name/password!'
    return render_template('login.html', msg='')


@app.route('/logout')
def logout():
   session['loggedin'] = False
   return render_template("login.html")

@app.route('/register', methods=['GET', 'POST'])
def register():  
    if request.method=='POST':
        userdata=request.form
        name=userdata['name']
        email=userdata['email']
        password=userdata['password']
        
        con=dbConnection()
        cursor=con.cursor()   
       
        cursor.execute("INSERT INTO user(name,email,password) VALUES(%s,%s,%s)",(name,email,password))
        con.commit()
       
        return  render_template("login.html")
    return render_template('register.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        f= request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        df=pd.read_csv(file_path)
        # print(df)

        df1 = df.drop(['STATEFIPS', 'agi_stub', 'STATE','zipcode','joint_return'], axis=1)
        df = df1.replace('', np.nan)
        dff = df.fillna(0)

        new_list=[]
        filename='svm_pickle'
        loaded_model_rf = pickle.load(open(filename, 'rb'))
        rf1=loaded_model_rf.predict(dff)
        output_svm=rf1[0]
        print("-------------------------------------------------------------------------------")
        print(output_svm)
        print("-------------------------------------------------------------------------------")
        if output_svm==1:
            new_list.append(output_svm)
            flash("Risk Detected")            
        else:
            new_list.append(output_svm)
            flash("Risk not Detected")

        dff["Risk"] = dff["return_with_capital_gain_loss"] * dff["capital_gain_loss"]
        dff["Risk"] = dff["Risk"]/1000000 
        dff.loc[dff.Risk >= 1, ['final_risk']] = 1
        dff['final_risk'] = dff['final_risk'].fillna(0)
        
        a=new_list.count(1)
        b=new_list.count(0)
        print("-------------------------------------------------------------------------------")

        if a>b:
            rt="Risk is there"  
        else:
            nr="Risk is not there"
#-----------------------------Ammendment Approach---------------------------------------------------------------------------------------------       
        x=dff['IRS_payment_amnt'][0]
        if a>b:
            if x >= 10000:
                flash("User is in risk found by Amendment Approach")
            else:
                flash("user is not in risk")
        elif nr=="Risk is not there":
            flash("user is not in Amendment Approach")
        else:
            flash("User not in risk")
            

        amnd_taxpayers = dff.iloc[(dff['IRS_payment_amnt'] > 100000).values]
       
        print("-------------------------------------------------------------------------------")
        print("Ammendment Approach:")
        print(amnd_taxpayers)
        print("-------------------------------------------------------------------------------")

#-----------------------------diagnostic Approach--------------------------------------------------------------------------------------------- 
        is_NaN = dff.isnull()
        row_has_NaN = is_NaN.any(axis=1)
        digno_taxpayers = dff[row_has_NaN]

        is_NaN = dff.isin([0]).sum()
        # print(type(is_NaN))
        count=0


       

        for i in range(len(is_NaN)):
            if is_NaN[i]==0:
                count+=1
            else:
                pass

        if a>b:
            if count>0:
                flash("User is in risk found by diagnostic Approach")
            else:
                flash("user not in risk")
        elif nr=="Risk is not there":
            flash("user is not in diagnostic Approach")
        else:
            flash("User not in risk")      
#-----------------------------Statement Approach---------------------------------------------------------------------------------------------- 
        Net_income = ((dff['IRS_payment_amnt'] + dff['capital_gain_loss']) - (dff['expenses_amount'] + dff['capital_gain_loss']))  
        dff['Net_income'] =  Net_income
        # if dff[(dff.Net_income <= 0)]:
        #     print("please correct your file")
       

        if a>b:
            if (dff['Net_income']).any() >= 100000:
                flash("User is in risk found by Statement Approach")
            else:
                flash("user not in risk")
        elif nr=="Risk is not there":
            flash("user is not in Statement Approach")
        else:
            flash("User not in risk") 

        stmnt_taxpayers = dff.iloc[(dff['taxable_intrest_amnt'] > 100000).values]
        
        print("Statment Approach:")
        print(stmnt_taxpayers)
#-----------------------------Volatility Approach---------------------------------------------------------------------------------------------- 
        # Compute the logarithmic returns using the Closing price 
        dff['Log_Ret'] = np.log(dff['Risk'] / dff['Risk'])

        dff['vola'] = dff['Log_Ret'].std()
        dff['vola'].fillna(dff['vola'].median(), inplace=True)
        dff['Log_Ret'].fillna(dff['Log_Ret'].median(), inplace=True)
        # df['vola'].gt(3).sum()

        if a>b:
            if (dff['vola']).any() >= 3:
                flash("User is in risk found by Volatility Approach")
            else:
                flash("user not in risk")
        elif nr=="Risk is not there":
            flash("user is not in Volatility Approach")
        else:
            flash("User not in risk") 

        Vol_taxpayers = dff.iloc[(dff['vola'] > 3.0).values]
        print("Volatility Approach:")
        print(Vol_taxpayers)
#-----------------------------colorful Approach---------------------------------------------------------------------------------------------- 
        a = stmnt_taxpayers.append(digno_taxpayers)
        b = a.append(amnd_taxpayers)
        fdf = b.append(Vol_taxpayers)
        print("high risk taxpayer found by Colorful Approach")
        print(fdf)
        
        return render_template('prediction.html', tables=[fdf.to_html(classes='data')], titles=fdf.columns.values)
    return render_template('prediction.html')
    
@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    # app.run('0.0.0.0')
    app.run(debug=True)