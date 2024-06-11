import numpy as np 
import pandas as pd 
import chart_studio.plotly  # Import chart-studio first
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
from datetime import datetime
import geopandas as gpd
from scipy import stats
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import zipfile
import requests
from io import BytesIO


# Set display options
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format

# Set the default renderer for plotly
#pio.renderers.default = 'browser'

# URL to the raw zip file in your GitHub repository
zip_url = 'https://github.com/AureleData/exploratory-data-analysis-on-retail-database/raw/main/online_retail.zip'

# Download the zip file
response = requests.get(zip_url)

# Read data from the zip file
with zipfile.ZipFile(BytesIO(response.content), 'r') as zip_ref:
    # Assuming the CSV file is named 'online_retail.csv' inside the zip archive
    with zip_ref.open('online_retail.csv') as csv_file:
        df = pd.read_csv(csv_file)

# Display the dataframe
print(df)


# Display data information
df.info()
# We take it in absolute values, as we can find if it is a cancellation using the C in front of invoiceNo
df["Quantity"] = abs(df["Quantity"])
df["UnitPrice"] = abs(df["UnitPrice"])

df1=df.copy()

# Check for NA values
df1.isna().sum()

# Some descriptions are missing and a good number of customerID as well

# Check duplicates
df1.duplicated().sum()

# For the current analysis, we decide to remove the missing values
df1.dropna(subset=['Description'], inplace=True)
df1.dropna(subset=['CustomerID'], inplace=True)

# Drop index column
df1 = df1.drop(columns=['index'])
# Then, convert the column to integers and then to strings
df1['CustomerID'] = df1['CustomerID'].astype(int).astype(str)

df1.info()

# Let's create a new column for the amount of sales
df1['Total_sales'] = df1['Quantity'] * df1['UnitPrice']
df1 = df1[df1['Total_sales'] != 0]

# Summary statistics
df1.describe()

# Average quantity sold is 12.06 while the average price is 3.46€, and the average total_sales is 20.4

# The minimum quantity is equal to the maximum one as it was probably a canceled order
# Minimum unit price is 0, we should have a look at this as it means an item is for free, while the max is 38970
# Total sales follow the same pattern as quantity which is logic as quantity and total sales are dependent on each other

# Let's have a look at the maximum, minimum etc
# Describe the minimum unit price
min_unit_price = df1['UnitPrice'].min()
min_unit_price_description = df1[df1['UnitPrice'] == min_unit_price]['Description'].iloc[0]
min_unit_price_description

# Minimum unit price is 'ROUND CAKE TIN VINTAGE GREEN' let's see if this item price is always 0
# Filter the DataFrame for rows where Description is 'ROUND CAKE TIN VINTAGE GREEN'
green_cake_rows = df1[df1['Description'] == 'ROUND CAKE TIN VINTAGE GREEN']

# Check if the unit price is always 0 for this description
is_always_zero = green_cake_rows['UnitPrice'].eq(0).all()
# Print the result
if is_always_zero:
    print("Unit price is always 0 when Description is 'ROUND CAKE TIN VINTAGE GREEN'")
else:
    print("Unit price is not always 0 when Description is 'ROUND CAKE TIN VINTAGE GREEN'")
# Ok so it seems it is not always 0, let's see why

# Filter the DataFrame for rows where Description is 'ROUND CAKE TIN VINTAGE GREEN'
green_cake_data = df1[df1['Description'] == 'ROUND CAKE TIN VINTAGE GREEN']

# Calculate the average, maximum, and minimum unit prices
average_unit_price = green_cake_data['UnitPrice'].mean()
max_unit_price = green_cake_data['UnitPrice'].max()
min_unit_price = green_cake_data['UnitPrice'].min()

# Print the results
print("Average Unit Price for 'ROUND CAKE TIN VINTAGE GREEN':", average_unit_price)
print("Maximum Unit Price for 'ROUND CAKE TIN VINTAGE GREEN':", max_unit_price)
print("Minimum Unit Price for 'ROUND CAKE TIN VINTAGE GREEN':", min_unit_price)

# So maximum price is 15.79 here, let's see what is the most common 0 or 15.79

# Filter the DataFrame for rows where Description is 'ROUND CAKE TIN VINTAGE GREEN'
green_cake_rows = df1[df1['Description'] == 'ROUND CAKE TIN VINTAGE GREEN']

# Get the unique unit prices
unique_unit_prices = green_cake_rows['UnitPrice'].unique()

# Print the unique unit prices
print("Unique Unit Prices for 'ROUND CAKE TIN VINTAGE GREEN':")
print(unique_unit_prices)

# There are 5 different unit prices for this item, let's see which one occurs the most often 

# Filter the DataFrame for rows where Description is 'ROUND CAKE TIN VINTAGE GREEN'
green_cake_rows = df1[df1['Description'] == 'ROUND CAKE TIN VINTAGE GREEN']

# Get the count of occurrences for each unique unit price
unit_price_counts = green_cake_rows['UnitPrice'].value_counts()

# Print the number of occurrences for each unique unit price
print("Number of Occurrences for each unique Unit Price for 'ROUND CAKE TIN VINTAGE GREEN':")
print(unit_price_counts)

# Ok, we have our answer, it is 7.95. We could dig deeper and see for which country and which customers these prices varied
# Also, we have only one entry when unitprice is 0 for this item, it raises the question to keep it for this one and the others
# Let's see what are the other data for this particular occurrence and for other items when price is 0

# Filter the DataFrame for rows where Description is 'ROUND CAKE TIN VINTAGE GREEN' and UnitPrice is 0
green_cake_zero_price_rows = df1[(df1['Description'] == 'ROUND CAKE TIN VINTAGE GREEN') & (df1['UnitPrice'] == 0)]
# Print the filtered data
print(green_cake_zero_price_rows)

# Looks like it could be an error in the data as only this occurrence has 0

# Filter the DataFrame for rows where UnitPrice is 0
zero_price_rows = df1[df1['UnitPrice'] == 0]
zero_price_rows

# Let's have a look at the max unit price now and see if it follows a similar pattern
# Describe the maximum unit price
max_unit_price = df1['UnitPrice'].max()
max_unit_price_description = df1[df1['UnitPrice'] == max_unit_price]['Description'].iloc[0]
max_unit_price_description

# Filter the DataFrame for rows where Description is 'Manual'
Manual_data = df1[df1['Description'] == 'Manual']

# Calculate the average, maximum, and minimum unit prices for 'Manual'
average_unit_price = Manual_data['UnitPrice'].mean()
max_unit_price = Manual_data['UnitPrice'].max()
min_unit_price = Manual_data['UnitPrice'].min()

# Print the results
print("Average Unit Price for 'Manual':", average_unit_price)
print("Maximum Unit Price for 'Manual':", max_unit_price)
print("Minimum Unit Price for 'Manual':", min_unit_price)

# Get the unique unit prices for 'Manual'
unique_unit_prices = Manual_data['UnitPrice'].unique()

# Print the unique unit prices
print("Unique Unit Prices for 'Manual':")
print(unique_unit_prices)

# There are many different unit prices for this item, let's see which one occurs the most often 
# Filter the DataFrame for rows where Description is 'Manual'
Manual_rows = df1[df1['Description'] == 'Manual']

# Get the count of occurrences for each unique unit price
unit_price_counts = Manual_rows['UnitPrice'].value_counts()

# Print the number of occurrences for each unique unit price
print("Number of Occurrences for each unique Unit Price for 'Manual':")
print(unit_price_counts)

# 1.25 seems to be the most common price and due to the variety of prices, we cannot decide to change anything

# Let's have a look at the maximum price
# Filter the DataFrame for rows where UnitPrice is 38970
max_price_rows = df1[df1['UnitPrice'] == 38970]
max_price_rows

# The 'C' in the invoiceno means it is a canceled order as confirmed by the negative quantity
# We know this df1 may have fraudulent data and this could very much be an example of that, considering that
# the cancel price should be equal to a bought price and there is no counterpart of positive quantity for the same price

# Let's have a look at the quantities now
# Describe the minimum unit price
min_quantity = df1['Quantity'].min()
min_quantity_description = df1[df1['Quantity'] == min_quantity]['Description'].iloc[0]
min_quantity_description

# We have 'PAPER CRAFT , LITTLE BIRDIE' as the minimum quantity
# Filter the DataFrame for rows where Description is 'PAPER CRAFT , LITTLE BIRDIE'
PAPER_CRAFT_data = df1[df1['Description'] == 'PAPER CRAFT , LITTLE BIRDIE']

# Calculate the average, maximum, and minimum unit prices for 'PAPER CRAFT , LITTLE BIRDIE'
average_unit_price = PAPER_CRAFT_data['Quantity'].mean()
max_unit_price = PAPER_CRAFT_data['Quantity'].max()
min_unit_price = PAPER_CRAFT_data['Quantity'].min()

# Print the results
print("Average Quantity for 'PAPER CRAFT , LITTLE BIRDIE':", average_unit_price)
print("Maximum Quantity for 'PAPER CRAFT , LITTLE BIRDIE':", max_unit_price)
print("Minimum Quantity for 'PAPER CRAFT , LITTLE BIRDIE':", min_unit_price)

# Get the unique quantities for 'PAPER CRAFT , LITTLE BIRDIE'
unique_unit_prices = PAPER_CRAFT_data['Quantity'].unique()

# Print the unique quantities
print("Unique Quantities for 'PAPER CRAFT , LITTLE BIRDIE':")
print(unique_unit_prices)

# These 2 quantities are the only 2 that exist for this particular item
# This is rather odd; it could be an error, someone ordered this by error and then canceled it

# Filter the DataFrame for rows where Quantity is 0
zero_quantity_rows = df1[df1['Quantity'] == 0]
zero_quantity_rows

# There are no entries with 0 in quantity, which makes sense but still, we better check it
# We could remove these 2 quantities from the df1. They don't bring much in terms of information except that 
# maybe someone did a bad manipulation and corrected it. The cancel price is the same as the order price
# It doesn't look like there is a fraudulent transaction here


# Data manipulation for visualization

# As we had this previous issue with unit price = 0
# Here we can see all the items with 0 as unit price, it seems these are errors
# We have 2 solutions: delete them would be the simplest one
# or we can replace the 0 by the average unit price by stock code, which is linked to description

# Calculating the average unit price for each stock code
average_unit_price_per_stockcode = df1.groupby('StockCode')['UnitPrice'].mean()

# Replace the unit price when it is 0 with the average unit price for the corresponding stock code
df1['UnitPrice'] = df1.apply(lambda row: average_unit_price_per_stockcode[row['StockCode']] if row['UnitPrice'] == 0 else row['UnitPrice'], axis=1)

# Let's have a look again
# Filter the DataFrame for rows where UnitPrice is 0
zero_price_rows = df1[df1['UnitPrice'] == 0]
zero_price_rows

# No more unit price = 0

# Filter the DataFrame for rows where CustomerID is 15098
customer_15098_rows = df1[df1['CustomerID'] == '15098']
customer_15098_rows

# Group the DataFrame by CustomerID, StockCode, UnitPrice, and Description
grouped = df1.groupby(['CustomerID', 'StockCode', 'UnitPrice', 'Description'])

# Filter out rows where the sum of the quantities is not equal to zero
filtered_df1 = grouped.filter(lambda x: x['Quantity'].sum() != 0)

# Print the filtered DataFrame
print(filtered_df1)

# This filtered_df1 contains only the entries which do not cancel each other directly within the same stock code
# In case we do not want to look at cancellation analysis here, if we want, we can skip the previous step
# We removed around 3k rows by doing this

# Summary
filtered_df1.describe()
filtered_df1.info()

# Some date transformation for time series analysis
filtered_df1['InvoiceDate']=pd.to_datetime(filtered_df1['InvoiceDate'])
filtered_df1['Invoice_year']=filtered_df1['InvoiceDate'].dt.year
filtered_df1['Invoice_month']=filtered_df1['InvoiceDate'].dt.month
filtered_df1['Invoice_hour']=filtered_df1['InvoiceDate'].dt.hour


#removing outliers by the average

# Function to calculate outlier thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return up_limit, low_limit

# Function to replace outliers with the average (rounded for Quantity)
def replace_with_average(dataframe, variable):
    up_limit, low_limit = outlier_thresholds(dataframe, variable)
    if variable == "Quantity":
        avg_value = round(dataframe[variable].mean())  # Round to the nearest whole number
    else:
        avg_value = dataframe[variable].mean()
    dataframe.loc[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit), variable] = avg_value

# Plot before replacing outliers
fig, axes = plt.subplots(2, 1, figsize=(15, 10))  # Adjusted figsize for better clarity
col_list = ["Quantity", "UnitPrice"]
for i in range(2):
    axes[i].boxplot(filtered_df1[col_list[i]], flierprops=dict(marker="o", markerfacecolor="blue"), vert=0)
    axes[i].set_title(col_list[i] + " (Before)")

plt.tight_layout()
plt.show()

# Replace outliers with the average
for col in col_list:
    replace_with_average(filtered_df1, col)

# Plot after replacing outliers
fig, axes = plt.subplots(2, 1, figsize=(15, 10))  # Adjusted figsize for better clarity
for i in range(2):
    axes[i].boxplot(filtered_df1[col_list[i]], flierprops=dict(marker="o", markerfacecolor="blue"), vert=0)
    axes[i].set_title(col_list[i] + " (After)")

plt.tight_layout()
plt.show()

#recalculate the Total_sales
filtered_df1['Total_sales'] = filtered_df1['Quantity'] * filtered_df1['UnitPrice']

####### Start of data visualization

# Average Sales per Country
sales_per_country = filtered_df1.groupby("Country")["Total_sales"].mean().sort_values(ascending=False)

# Using "plasma" palette and reversing it
palette = sns.color_palette("plasma", len(sales_per_country))[::-1]

plt.figure(figsize=(12, 7), dpi=150)
sns.barplot(x=sales_per_country.values, y=sales_per_country.index, palette=palette)  
plt.xlabel("Average Sales")
plt.ylabel("Country")
plt.title("Average Sales per Country")
plt.show()

# Average quantity by country
avg_quantity_by_country = filtered_df1.groupby('Country')['Quantity'].mean().sort_values(ascending=False)

# Visualization with "viridis" palette
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_quantity_by_country.values, y=avg_quantity_by_country.index, palette='viridis')
plt.title('Average Quantity Sold in Each Country')
plt.xlabel('Average Quantity Sold')
plt.ylabel('Country')
plt.show()

# The frequencies of clients by country

# Cross-tabulation of customers by country with normalization by columns (to get percentages)
country_freq_percent = pd.crosstab(index=filtered_df1['Country'], columns='count', normalize='columns')

# Sort the cross-tabulation by frequency in descending order
country_freq_percent_sorted = country_freq_percent.sort_values(by='count', ascending=False)

# Group countries with less than 1.6% into a single category
threshold = 0.016
small_countries = country_freq_percent_sorted[country_freq_percent_sorted['count'] < threshold].index
other_percentage = country_freq_percent_sorted[country_freq_percent_sorted['count'] < threshold]['count'].sum()
country_freq_percent_sorted.drop(small_countries, inplace=True)

# Calculate percentage of total for the "Other" category
total_percentage = 1 - country_freq_percent_sorted['count'].sum()
country_freq_percent_sorted.loc['Other'] = total_percentage

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(country_freq_percent_sorted['count'] * 100, labels=country_freq_percent_sorted.index, autopct='%1.1f%%', startangle=140)
plt.title('Clients by Country')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show plot
plt.show()

# Very much predominantly clients from UK, then around 2% for Germany, France, and EIRE.
# Then all the other countries together do not even add up to 5%.

# This is confirmed by the top sales
top_countries = filtered_df1.groupby('Country')['Total_sales'].sum().sort_values(ascending=False).head(10)

# Visualization
plt.figure(figsize=(16, 6))
top_countries.plot(kind='bar', color='steelblue')
plt.title('Top 10 Countries by Sales')
plt.xlabel('Country')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Total Sales')
plt.show()



######################## Let's have a look at sales in values and quantity

# Sales by month
# Calculate total sales by invoice month
total_sales_by_month = filtered_df1.groupby('Invoice_month')['Total_sales'].sum()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=total_sales_by_month.index, y=total_sales_by_month.values, palette='muted')
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# It seems that there is a spike in sales between August to November, which makes sense as preparations for Christmas
# with a big spike in November.

# Average Sales per Month
per_month = filtered_df1.groupby("Invoice_month")["Total_sales"].mean()

plt.figure(dpi=100)
sns.barplot(x=per_month.index, y=per_month.values, palette='deep')
plt.ylabel("Average Total Sales")
plt.title("Average Sales Per Month")
plt.show()

# This graph tells us that the average of total sales stays between 15 to 20 all year long.
# There is even a small dip in November, we can assume that people spend a bit less per order
# as they do more orders. Let's confirm by looking at the average quantity order by month.

# Calculate average quantity by invoice month
avg_quantity_by_month = filtered_df1.groupby('Invoice_month')['Quantity'].mean()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_quantity_by_month.index, y=avg_quantity_by_month.values, palette='muted')
plt.title('Average Quantity by Month')
plt.xlabel('Month')
plt.ylabel('Average Quantity')
plt.show()

# Let's see total quantity
# Calculate total quantity by invoice month
total_quantity_by_month = filtered_df1.groupby('Invoice_month')['Quantity'].sum()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=total_quantity_by_month.index, y=total_quantity_by_month.values, palette='muted')
plt.title('Total Quantity by Month')
plt.xlabel('Month')
plt.ylabel('Total Quantity')
plt.show()

# Let's see if it correlates with the number of orders, we should see a spike in the number of orders between August and November
# Count the number of unique InvoiceNo by invoice month
invoice_count_by_month = filtered_df1.groupby('Invoice_month')['InvoiceNo'].nunique()

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=invoice_count_by_month.index, y=invoice_count_by_month.values, palette='muted')
plt.title('Number of InvoiceNo by Month')
plt.xlabel('Month')
plt.ylabel('Number of InvoiceNo')
plt.show()

# So it tells us that even if the quantity and total_sales do not vary much over the year, there is a big spike
# in the number of orders between August and November.



############# To Conclude

# What it tells us, is that people spend less and buy less quantity per order between August and November.
# However, there is also much more orders over this period which lead to an overall increase in sales, both in values and quantity.
# Here is the confirmation of that:

# Monthly Revenue Trend
monthly_revenue = filtered_df1.resample('M', on='InvoiceDate')['Total_sales'].sum()

# Visualization
plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind='line', marker='o', color='blue')
plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.show()



# Monthly Quantity Trend
monthly_quantity = filtered_df1.resample('M', on='InvoiceDate')['Quantity'].sum()

# Visualization
plt.figure(figsize=(12, 6))
monthly_quantity.plot(kind='line', marker='o', color='blue')
plt.title('Monthly Quantity Trend')
plt.xlabel('Month')
plt.ylabel('Total Quantity')
plt.show()

# Let's look at the frequency of quantity per country

# Calculate total quantity
total_quantity = filtered_df1['Quantity'].sum()

# Calculate quantity percentage by country
quantity_percentage_by_country = (filtered_df1.groupby('Country')['Quantity'].sum() / total_quantity) * 100

# Plotting a pie chart
plt.figure(figsize=(10, 8))
plt.pie(quantity_percentage_by_country, labels=quantity_percentage_by_country.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage of Total Quantity by Country')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# UK represents obviously 81.4% of all the quantity sold.
# What if we remove UK to have a better view at other countries?

#### If we look at the data without UK
# Filter out United Kingdom data
filtered_df_without_uk = filtered_df1[filtered_df1['Country'] != 'United Kingdom']

# Calculate total quantity excluding United Kingdom
total_quantity_without_uk = filtered_df_without_uk['Quantity'].sum()

# Calculate quantity percentage by country excluding United Kingdom
quantity_percentage_by_country_without_uk = (filtered_df_without_uk.groupby('Country')['Quantity'].sum() / total_quantity_without_uk) * 100

# Group countries with less than 1% of total quantity
mask = quantity_percentage_by_country_without_uk < 1
other_countries_quantity = quantity_percentage_by_country_without_uk[mask].sum()

# Filter out countries with less than 1% of total quantity
quantity_percentage_by_country_filtered = quantity_percentage_by_country_without_uk[~mask]

# Group the rest as 'Other'
quantity_percentage_by_country_filtered['Other'] = other_countries_quantity

# Plotting a pie chart
plt.figure(figsize=(10, 8))
plt.pie(quantity_percentage_by_country_filtered, labels=quantity_percentage_by_country_filtered.index, autopct='%1.1f%%', startangle=140)
plt.title('Percentage of Total Quantity by Country (Excluding United Kingdom)')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Clients from Netherlands, Germany, Eire, and France are buying the most in terms of quantity if we remove UK. It is much more spread.

# Now let's have a look at the best sellers, first in terms of sales value and then in terms of quantity and see if they are different.
# We need to remove the invoiceNo starting with C as they are not sales obviously.

# Filter out rows where InvoiceNo starts with "C"
filtered_df_no_cancellation = filtered_df1[~filtered_df1['InvoiceNo'].str.startswith('C')]

# Calculate total sales for each description
total_sales_by_description = filtered_df_no_cancellation.groupby('Description')['Total_sales'].sum()

# Sort the descriptions by total sales in descending order and get the top 10
top_10_descriptions = total_sales_by_description.sort_values(ascending=False).head(10)

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_descriptions.values, y=top_10_descriptions.index, palette='muted')
plt.title('Top 10 Items by Total Sales (Excluding Cancellations)')
plt.xlabel('Total Sales')
plt.ylabel('Item')
plt.yticks(rotation=45)
plt.gca().tick_params(axis='y', labelsize=8)
plt.show()

# Top sale in value is Regency Cakestand 3 tier.

# Now by quantity

# Filter out rows where InvoiceNo starts with "C"
filtered_df_no_cancellation = filtered_df1[~filtered_df1['InvoiceNo'].str.startswith('C')]

# Calculate total quantity for each description
total_quantity_by_description = filtered_df_no_cancellation.groupby('Description')['Quantity'].sum()

# Sort the descriptions by total quantity in descending order and get the top 10
top_10_descriptions = total_quantity_by_description.sort_values(ascending=False).head(10)

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_descriptions.values, y=top_10_descriptions.index, palette='muted')
plt.title('Top 10 Items by Total Quantity (Excluding Cancellations)')
plt.xlabel('Total Quantity')
plt.ylabel('Item')
plt.yticks(rotation=45)
plt.gca().tick_params(axis='y', labelsize=8)
plt.show()

# Top sales in quantity is Jumbo bag red Retrospot.

# Now we will look at the sales distribution of the top 1 best seller by sales and quantity over time.

# Find the top sales item by total sales
top_sales_item = total_sales_by_description.idxmax()

# Grouping by year and month and summing the total sales for the top sales item
top_sales_item_data_grouped = filtered_df1[filtered_df1['Description'] == top_sales_item].groupby(['Invoice_year', 'Invoice_month'])['Total_sales'].sum()

# Plotting the time distribution of total sales for the top sales item by year and month
plt.figure(figsize=(12, 6))
top_sales_item_data_grouped.plot(kind='line', marker='o', color='skyblue')
plt.title(f'Time Distribution of Total Sales for "{top_sales_item}" by Year and Month')
plt.xlabel('Year-Month')
plt.ylabel('Total Sales')

# Set more ticks on the x-axis
plt.xticks(range(len(top_sales_item_data_grouped.index)), top_sales_item_data_grouped.index, rotation=45)

plt.grid(axis='y')
plt.show()

# Spike of sales for the top items by total_sales in March 2011.

# Let's see if the top sold item by quantity followed the same pattern or not.

# Find the top sales item by quantity
top_sales_item_by_quantity = total_quantity_by_description.idxmax()

# Grouping by year and month and summing the total quantity for the top sales item
top_sales_item_quantity_grouped = filtered_df1[filtered_df1['Description'] == top_sales_item_by_quantity].groupby(['Invoice_year', 'Invoice_month'])['Quantity'].sum()

# Plotting the time distribution of total quantity for the top sales item by year and month
plt.figure(figsize=(12, 6))
top_sales_item_quantity_grouped.plot(kind='line', marker='o', color='skyblue')
plt.title(f'Time Distribution of Total Quantity for "{top_sales_item_by_quantity}" by Year and Month')
plt.xlabel('Year-Month')
plt.ylabel('Total Quantity')

# Set more ticks on the x-axis
plt.xticks(range(len(top_sales_item_quantity_grouped.index)), top_sales_item_quantity_grouped.index, rotation=45)

plt.grid(axis='y')
plt.show()

# Top quantity sold was reached on August 2011 for Jumbo bag red retrospot.





#### data management for transformation for powerBI

#stockcode and description checks
# Group by stockcode and count unique descriptions
description_count = filtered_df1.groupby('StockCode')['Description'].nunique()

# Filter to find stock codes with more than one unique description
multiple_descriptions = description_count[description_count > 1]

print(multiple_descriptions)


df_powerbi=filtered_df1.copy()



# Identify the most common description for each stock code
description_counts = df_powerbi.groupby(['StockCode', 'Description']).size().reset_index(name='count')

# Identify the most common description for each stockcode
most_common_descriptions = description_counts.loc[description_counts.groupby('StockCode')['count'].idxmax()]


# Create a dictionary to map stockcodes to their most common descriptions
most_common_dict = most_common_descriptions.set_index('StockCode')['Description'].to_dict()

# Replace descriptions with the most common description for each stockcode
df_powerbi['Description'] = df_powerbi['StockCode'].map(most_common_dict)

print(df_powerbi[['StockCode', 'Description']])


#stockcode and description checks
# Group by stockcode and count unique descriptions
description_count = df_powerbi.groupby('StockCode')['Description'].nunique()

# Filter to find stock codes with more than one unique description
multiple_descriptions = description_count[description_count > 1]

print(multiple_descriptions)


# Group by description and count unique stock codes
stockcode_counts = df_powerbi.groupby('Description')['StockCode'].nunique().reset_index()

# Filter to find descriptions with more than one unique stock code
descriptions_with_multiple_stockcodes = stockcode_counts[stockcode_counts['StockCode'] > 1]

# Display the result
print(descriptions_with_multiple_stockcodes)

#export to CSV
df_powerbi.to_csv('treated_retail.csv', sep=';', index=False)