
"""
Created on Tue Mar 24 16:46:11 2020
For CS526 Chicago Crime Data Visualisation

@author: srikaavya
"""
import pandas as pd
import numpy as np
import json
import datetime
import os
file_name=""        # for storing the filename
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename
from fbprophet import Prophet

app = Flask(__name__)
use_cols=['Date','Domestic','PrimaryType','District','Arrest','Year','FBICode','LocationDescription']

# reading data from file
data = pd.read_csv('/Users/srikaavya/Downloads/CrimeData.csv', usecols=use_cols, iterator=True, chunksize=100000)
data = pd.concat(data, ignore_index=True)
dist_dict={1:"Central", 2:"Wentworth", 3:"Grand Crossing", 4:"South Chicago", 5:"Calumet", 6:"Gresham",
            7:"Eaglewood", 8:"Chicago Lawn", 9:"Deering", 10:"Ogden", 11:"Harrison", 12:"Near West",
           14:"Shakespeare", 15:"Austin", 16:"Jefferson Park", 17:"Albany Park", 18:"Near North",
           19:"Town Hall", 20:"Lincoln", 22:"Morgan Park", 24:"Rogers Park", 25:"Grand Central"}  #district names
rm_dist = {1,2,3,4,5,6,7,8,9,10,11,12,14,15,16,17,18,19,20,22,24,25} # district numbers
data = data[data.District.isin(rm_dist)]
districts = data['District'].unique()
data.Date = data['Date'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y %I:%M:%S %p"))
crimes = data['PrimaryType'].unique()

#print(data.head())
#-----------------------------------------------
#   methods for data processing and visualisation
#-----------------------------------------------

def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj,datetime.datetime):
            return obj.__str__()

# Visualisation for pie chart
# ---------------------------------------------------------
# function  pie_chart
# inputs: Dataframe
# Output: dictionary with data converted to json format
#-----------------------------------------------------------
def pie_chart():
    pie_chart_data = data['PrimaryType'].value_counts().rename_axis('Crime').reset_index(name='counts')
    return pie_chart_data

# Visualisation for pie of pie chart
# ---------------------------------------------------------
# function pie_of_pie
# inputs: Dataframe
# Output: dictionary with data converted to json format
#-----------------------------------------------------------
@app.route('/pie_of_pie')
def pie_of_pie():
    domestic = data[data['Domestic'] == True]   #filtering for domestic crimes
    slice_pie_data = domestic[['PrimaryType', 'District']]
    crimes = slice_pie_data.groupby(['PrimaryType']).size().reset_index(name='counts')
    max_crimes = crimes.counts.nlargest(15).iloc[-1]
    slice_pie_data = slice_pie_data.groupby(['PrimaryType', 'District']).size().reset_index(name='counts')
    dict = []
    list = {}
    for crime in  slice_pie_data["PrimaryType"].unique():
        c = domestic[domestic.PrimaryType == crime]['PrimaryType'].count()
        if c > max_crimes:
            filtered = slice_pie_data[slice_pie_data.PrimaryType == crime]
            list['Crime'] = crime
            list['Count'] = c
            list['subData'] = []
            sub_list = {}
            max_limit = filtered.counts.nlargest(10).iloc[-1]
            for row in filtered.itertuples(index=False):
                if  row[2] > max_limit:
                    sub_list["District"] = dist_dict[row[1]]
                    sub_list["value"] = row[2]
                    list['subData'].append(sub_list)
                    sub_list = {}
            dict.append(list)
            list = {}
    pie = json.dumps(dict, default=myconverter)
    return pie

# Visualisation for pie  chart
# ---------------------------------------------------------
# function Pie_charts
# inputs: Dataframe
# Output: json output of records
#-----------------------------------------------------------
@app.route('/pie_charts')
def pie_charts():
    data=pie_chart().to_json(orient='records')
    return data

# Visualisation for bar race chart
# ---------------------------------------------------------
# function Bar_race
# inputs: Dataframe
# Output: dictionary with data converted to json format
#-----------------------------------------------------------
@app.route('/Bar_race')
def bar_race():
    race_data = data[['Year', 'District']]
    race_data = race_data.groupby(['Year', 'District']).size().reset_index(name='counts')
    mydict = {}     # creating an empty list to store the data
    for row in race_data.itertuples(index=True):
        if row.District not in mydict.keys():
            mydict[row.District] = row.counts
        else:
            mydict[row.District] = mydict[row.District] + row.counts
            race_data.at[row.Index, 'counts'] = mydict[row.District]
    race_dict = race_data.to_dict('records')
    for dist in race_dict:
        dist["District"] = dist_dict[dist["District"]] #storing the values with the corresponding
                                                        # district names.
    return jsonify(race_dict)

# Visualisation for Dumbell chart
# ---------------------------------------------------------
# function  Dumbell plot
# inputs: Dataframe
# Output: Dumbell data in dictionary in json format
#-----------------------------------------------------------
@app.route('/Dumbellplot')
def dumbell_plot():
    global data
    dumb_data = data[['PrimaryType', 'Arrest']]
    data = dumb_data.groupby(['PrimaryType', 'Arrest']).size().reset_index(name='counts')
    data = data.sort_values(["PrimaryType", "counts"], ascending = [True,False])

    # creating an empty dictionary and list to store the processed data
    dict = []
    list = {}
    for row in data.itertuples(index=False):
        if row[0] not in list.values():
            if row[2] < 20 :
                continue
            list['category'] = row[0]
            list['close'] = row[2]

        else:
            list['open'] = row[2]
            # append to the main list
            dict.append(list)
            list = {}           # clearing the list after appending to the dictionary
    dumbel_data=jsonify(dict)  # converting to json data
    return dumbel_data

# For safety rating (guage chart)
# ---------------------------------------------------------
# function safety_rating
# inputs: Dataframe
# Output: final_dict dictionary with data converted to json format
#-----------------------------------------------------------
def safety_rating(data):
    risk_categories = {'High': {'01A', '02', '04A', '04B'}, 'Moderate': {'01B', '03', '05', '06', '07', '09', '17'},
                       'Low': {'08B', '08A', '10', '11', '12', '13', '15', '16', '18', '19', '20', '22', '24', '26'}}
    rating_data = data[['District', 'PrimaryType', 'FBICode','Year']]

    # data is filtered based on the last year data
    rating_data = rating_data[rating_data.Year == 2020]

    rating_data = rating_data.groupby([ 'District', 'FBICode']).size().reset_index(name='counts')
    rating_data[rating_data['FBICode'].astype(str).str.isdigit()]

    dict2={} # creating an empty dictionary to store the data

    for data in rating_data.itertuples():
        if data.District not in dict2:
            risk_points = 0
            dict2[data.District] = risk_points

        if data[2] in risk_categories['High']:
            risk_points = risk_points + data[3]*2
        if data[2] in risk_categories['Moderate']:
            risk_points = risk_points + data[3]*1.5
        else:
            risk_points = risk_points + data[3]*1
        dict2[data.District]= int(risk_points/500)

    final_dict = {dist_dict[k]:v for k,v in dict2.items()}
    return final_dict

# Visualisation for guage chart
# ---------------------------------------------------------
# function guage_safety
# inputs: None
# Output: output of safety_rating in json format to provide
#           data in the specified url
#-----------------------------------------------------------

@ app.route('/Safety_Rating')
def gauge_safety():
    return jsonify(safety_rating(data))

# Visualisation for Word Cloud
# ---------------------------------------------------------
# function WordCloud
# inputs: None
# Output: Data for wordcloud in the specified URL
#-----------------------------------------------------------
@app.route('/WordCloud')
def WordCloud():
    data_wc = data['PrimaryType'].value_counts().rename_axis('tag').reset_index(name='counts')
    WordCloud_data = data_wc.to_json(orient='records')
    return WordCloud_data

#parameters for plotting the charts with new files
#   for dynamic plotting.
UPLOAD_FOLDER = '/Users/srikaavya/Downloads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'csv'}


# ---------------------------------------------------------
# function Upload_file()
# inputs: None
# Output: Html page redirected  for file upload
#-----------------------------------------------------------
@app.route('/Upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            global file_name
            file_name = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template('upload.html')


@app.route('/MorphPie', methods=['GET', 'POST'])
def MorphPie():
    pie_data = data[['District', 'LocationDescription']]
    pie_data = pie_data.groupby(['District', 'LocationDescription']).size().reset_index(name='counts')
    dict = []
    list = {}
    print("2")
    for crime in pie_data["District"].unique():
        c = pie_data[pie_data.District == crime]['District'].count()
        filtered = pie_data[pie_data.District == crime]
        list['Country'] = crime
        list['Count'] = c
        list['subData'] = []
        sub_list = {}
        max_limit = filtered.counts.nlargest(7).iloc[-1]
        for row in filtered.itertuples(index=False):
            if  row[2] > max_limit:
                sub_list["LocationDescription"] = row[1]
                sub_list["value"] = row[2]
                list['subData'].append(sub_list)
                sub_list = {}
        dict.append(list)
        list = {}
    fdata = json.dumps(dict, default=myconverter)
    return fdata

# ---------------------------------------------------------
# function allowed_file
# inputs: filename
# Output: None. TO check the filename has valid extensions
#-----------------------------------------------------------
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------
# function uploaded_file
# inputs: None
# Output: render template with the data to be shown in webpage
#-----------------------------------------------------------
@app.route('/plots',methods=['GET','POST'])
def uploaded_file():
    # process_file(filename)
    global file_name
    df = pd.read_csv(UPLOAD_FOLDER + file_name)
    col_type = df.columns.to_series().groupby(df.dtypes).groups
    # columns = list(df.columns)
    col_d = {k.name:v  for k,v in col_type.items()}
    col_int=[]
    col_obj=[]
    # print(col_obj)
    for key in col_d:
        if key == "int64":
            col_int.extend(col_d[key])
        if key == "object":
            col_obj.extend(col_d[key])
        # print(data_pl)
    print(col_int, col_obj)
    if request.method == "POST":
        plot_json = plots()
        print(plot_json)
        # plot_json = json.dumps(plot_json)
        # print(plot_json)
        return render_template('plots.html',data_x=col_int, data_y= col_obj, plot_data= plot_json)
    del df
    return render_template('plots.html',data_x=col_int, data_y= col_obj, plot_data="")

# ---------------------------------------------------------
# function plots
# inputs: None
# Output: returning the data required for plotting the new files
#-----------------------------------------------------------
def plots():
    df = pd.read_csv(UPLOAD_FOLDER + file_name)
    if request.method == "POST":
        x_col = request.form['drop_x']
        # print(x_col,"x_col")
        y_col = request.form['drop_y']
        # print(y_col,"y_col")
        data_pl = df[[x_col,y_col]]
        data_pl= data_pl.groupby(x_col).size().reset_index(name='counts')
        # print(data_pl)
        list = []
        temp_dict = {}
        for temp in data_pl.itertuples():
            temp_dict["name"] = str(temp[1])
            temp_dict["value"] = temp[2]
            # print(temp)
            list.append(temp_dict)
            temp_dict={}

        return list

# ---------------------------------------------------------
# function main_page
# inputs: None
# Output: Render template
#-----------------------------------------------------------
@app.route('/')
def main_page():
    return render_template('DIVA.html')

# ---------------------------------------------------------
# function Stats
# inputs: None
# Output: Render template
#-----------------------------------------------------------
@app.route('/Stats')
def Stats():
    return render_template('StatisticsView1.html')


# ---------------------------------------------------------
# function Domestic
# inputs: None
# Output: Render template
# -----------------------------------------------------------
@app.route('/Domestic')
def Domestic():
    return render_template('domestic.html')

# ---------------------------------------------------------
# function Judiciary()
# inputs: None
# Output: Render template
#-----------------------------------------------------------
@app.route('/Judiciary')
def Judiciary():
    return render_template('Judiciary.html')


# Run the main application and start the instance
if __name__ == "__main__":
    app.debug = True
    app.run()

