# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")


import pandas as pd
import datetime

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()

# Print summary statistics in next line
home_data.describe()

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data.LotArea.mean())

# Get the current date's information
now = datetime.datetime.now()

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = now.year - home_data.YearBuilt.max()

# Checks your answers
step_2.check()

''' Think about your data.

Notice that the latest data for YrSold is exactly 2010 as well; it seems unlikely that given all these houses,
not a single one was sold after 2010 (which also turns out to be the last recorded date for the house built.

This is a strong indication that the data collected is simply old (no data after 2010 is collected).

Expanding upon that, if we plot the histogram describing the distribution of when houses were being sold,
we will notice that 2006-2009 were years with roughly the same number of houses being sold (around 300),
while year 2010, there are only around 150-200 houses being sold.  

Furthermore, we can see roughly the same pattern in terms of when houses are being bought during the year
(a high around spring - April to July and lower in Fall-Winter).  However, upon inspection of 2010, it
seems like there is no data for after July - indicating that there is simply no data after July 2010'''


data_after_year_two_thousand = home_data[home_data.YrSold >= 2000]
#hist_year_built = data_after_year_two_thousand.YearBuilt.hist(bins=10)

#hist_year_sold = data_after_year_two_thousand.YrSold.hist(bins=10)

year = 2010
data_during_year = home_data[home_data.YrSold == year]

#hist_month_sold = data_during_year.MoSold.hist(bins=12)