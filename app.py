import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
import csv
# stream lit
import streamlit as st
from PIL import Image
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
#scores


st.sidebar.image('11697-removebg-preview.png', width=200)
st.header("Grocery Sale Prediction Application")


# Load your image
image = Image.open("supermarket-4052658__340.jpg")

# Display the image
st.image(image, caption="Your Image Caption")

df = pd.read_excel("items-2022-02-08-2023-02-09.xlsx", usecols=['Date', 'Time', 'Category','Item','Qty','PricePointName','SKU','GrossSales'])




st.sidebar.title("File Selection")
uploaded_file = st.sidebar.file_uploader("Upload a file")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df1 = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    elif uploaded_file.name.endswith('.xlsx'):
        df1 = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.json'):
        df1 = pd.read_json(uploaded_file)
    else:
        # handle other file formats
        pass
    
    merged_df = pd.concat([df1, df]) # create a list of dataframes to pass as the first argument to pd.concat()

  





# Data Load

# Change Datatype
df['Date'] = pd.to_datetime(df['Date'])

# split date into month day and year column
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year


# After spliting drop Time column
df.drop(['Time',"SKU","GrossSales"],axis=1,inplace=True)

# Drop Null Values
df.dropna(inplace=True)

# Manual label encoding of all category values
df['Category'] = df['Category'].replace({'Coffee & Tea': 0,'Bakery & Dessert':1, 'Beverages Taxable':2,
       'Breakfast Taxable':3, 'Lunch Taxable':4, 'Beer':5,
       'Grocery Non Taxable':6, 'Fruit Bunch':7, 'Grocery Taxable':8,
       'Soup & Crock':9, 'Frozen':10, 'Fruit Single':11, 'Bulk Snacks':12, 'Dairy':13,
       'No Barcode':14, 'Fine Foods & Cheese':15, 'Drug Store':16, 'Grab & Go':17,
       'Bread Retail':18, 'Candy':19, 'Chips & Snacks':20, 'Wine':21, 'Cigarettes':22,
       'Beverages Non Taxable':23, 'Produce':24, 'Wine No Barcode':25,
       'Health & Beauty':26, 'Hardware':27, 'None':28, 'Tobacco':29, 'Housewares':30,
       'Meat & Seafood':31, 'Full Meals Non Taxable':32, 'Paradise Remedies':33,
       'Swag':34, 'Mead':35, 'Gift Wrap':36, 'Beer No Barcode':37, 'Beer Single':38,
       'Holiday':39})

# Manual label encoding of all PricePointName values
le = LabelEncoder()
df['PricePointName'] = le.fit_transform(df['PricePointName'])
l = le.inverse_transform(df['PricePointName'])  
m=df.PricePointName.tolist()

size = pd.DataFrame(list(zip(m, l)),
               columns =['0', '1'])

size.to_csv("updated_size_labelencoding.csv",index=False)

# label encoding for Item column
df['Item'] = le.fit_transform(df['Item'])

# Inverse transform</h1>
l = le.inverse_transform(df['Item'])  
m=df.Item.tolist()
Item = pd.DataFrame(list(zip(m, l)),
               columns =['0', '1'])

Item.to_csv("updated_Item_labelencoding.csv",index=False)
dict = {'Coffee & Tea': 0,'Bakery & Dessert':1, 'Beverages Taxable':2,
       'Breakfast Taxable':3, 'Lunch Taxable':4, 'Beer':5,
       'Grocery Non Taxable':6, 'Fruit Bunch':7, 'Grocery Taxable':8,
       'Soup & Crock':9, 'Frozen':10, 'Fruit Single':11, 'Bulk Snacks':12, 'Dairy':13,
       'No Barcode':14, 'Fine Foods & Cheese':15, 'Drug Store':16, 'Grab & Go':17,
       'Bread Retail':18, 'Candy':19, 'Chips & Snacks':20, 'Wine':21, 'Cigarettes':22,
       'Beverages Non Taxable':23, 'Produce':24, 'Wine No Barcode':25,
       'Health & Beauty':26, 'Hardware':27, 'None':28, 'Tobacco':29, 'Housewares':30,
       'Meat & Seafood':31, 'Full Meals Non Taxable':32, 'Paradise Remedies':33,
       'Swag':34, 'Mead':35, 'Gift Wrap':36, 'Beer No Barcode':37, 'Beer Single':38,
       'Holiday':39}
# Save Data Frame
pd.DataFrame.from_dict(dict, orient='index').to_csv('updated_Category.csv')

# Model Implementation</h1>
df["Category"]=pd.to_numeric(df["Category"], errors='coerce')
df["Item"]=pd.to_numeric(df["Item"], errors='coerce')



# Drop Time And Implement Model
df = df.groupby(['Date', 'Category', 'Item','PricePointName']).agg({'Qty': 'sum'}).reset_index()

# Sort the data by date, category, and item
df = df.sort_values(['Date', 'Category', 'Item','PricePointName'])

# Set the date column as the index


# Check the resulting dataframe



# split date into month day and year column
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Year'] = df['Date'].dt.year

df.drop(['Date'],axis=1,inplace=True)

X = df.drop('Qty', axis=1)
y = df['Qty']


# Split Train Test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle =True)
model_ex= ExtraTreesRegressor(criterion= 'squared_error', max_features= None, random_state=1).fit( X_train, y_train)
y_prediction_ex = model_ex.predict(X_test)



# Define the number of days to predict ahead
num_days = 7

# Get the current date
current_date = datetime.today()

# Create a list of dates for the next week
date_list = [current_date + timedelta(days=x) for x in range(num_days)]

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(columns=['Day', 'Month', 'Year', 'Category', 'Item', 'PricePointName', 'Qty'])

# Loop through the dates and make predictions for each item and price point
for date in date_list:
    for category in df['Category'].unique():
        for item in df[df['Category'] == category]['Item'].unique():
            for price_point in df[(df['Category'] == category) & (df['Item'] == item)]['PricePointName'].unique():
                
                # Create a row of data to pass to the model
                row = {
                    'Day': date.day,
                    'Month': date.month,
                    'Year': date.year,
                    'Category': category,
                    'Item': item,
                    'PricePointName': price_point,
                    # 'Size': df1[(df1['Category'] == category) & (df1['Item'] == item) & (df1['PricePointName'] == price_point)]['Size'].iloc[0]
                }
                
                # Make a prediction using your trained model

                prediction = model_ex.predict([list(row.values())])[0]
                
                # Add the prediction to the DataFrame
                predictions_df = predictions_df.append({
                    'Day': row['Day'],
                    'Month': row['Month'],
                    'Year': row['Year'],
                    'Category': row['Category'],
                    'Item': row['Item'],
                    'PricePointName': row['PricePointName'],
                    'Qty': prediction
                }, ignore_index=True)

# Check the predictions
# predictions_df['Day'].describe()

category_df= pd.read_csv("updated_Category.csv")


item_df = pd.read_csv("updated_Item_labelencoding.csv")


Pricepointname_df = pd.read_csv("updated_size_labelencoding.csv")


# convert predictions_df into dataFrame
predictions_df = pd.DataFrame(predictions_df)

    
# Category_Original 
# change places of number and valuies column with each other
# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_Category.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["Category"])
original_values1 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Item_Original
# change places of number and valuies column with each other
item_df = item_df[['1', '0']]
# save df to csv
item_df.to_csv('updated_Item_labelencoding.csv', index=False)

# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_Item_labelencoding.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["Item"])
original_values2 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Price Point Name Original
# change places of number and valuies column with each other
Pricepointname_df = Pricepointname_df[['1', '0']]
# save df to csv
Pricepointname_df.to_csv('updated_size_labelencoding.csv', index=False)


# Load the label encoding CSV file into a dictionary object
label_encoding = {}
with open('updated_size_labelencoding.csv', mode='r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        label_encoding[int(row[1])] = row[0]

# Convert the encoded values back to the original categorical values
encoded_values = (predictions_df["PricePointName"])
original_values3 = [label_encoding[encoded_value] for encoded_value in encoded_values]


# Create the dataframe
final_df = pd.DataFrame({
    'Day': predictions_df['Day'],
    'Month': predictions_df['Month'],
    'Year': predictions_df['Year'],
    'Category': original_values1,
    'Item': original_values2,
    'Size': original_values3,
    'Qty': predictions_df['Qty'],
})


st.header("Prediction Result")


st.write(final_df)

# add a button to save the DataFrame to a file
if st.button('Download DataFrame'):
    # convert the DataFrame to a CSV string
    csv = final_df.to_csv(index=False)
    # use the file_downloader function to download the CSV string as a file
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='my_dataframe.csv',
        mime='text/csv'
    )

# create a search field
search_term = st.text_input('Search for an item')

    # filter the data based on the search term
if search_term:
    filtered_df = final_df[final_df['Item'].str.contains(search_term, case=False)]
else:
    filtered_df = final_df

    # display the filtered data
st.write(filtered_df)
