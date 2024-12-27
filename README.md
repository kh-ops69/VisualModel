This is a streamlit app that can be run both on your local machine and Google Colab. 
The dataset used here is sourced from kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download&select=fraudTest.csv

The above dataset is a Credit Card Fraud Detection Dataset and has the following description:
The simulator has certain pre-defined list of merchants, customers and transaction categories. And then using a python library called "faker", and with the number of customers, merchants that you mention during simulation, an intermediate list is created.

After this, depending on the profile you choose for e.g. "adults 2550 female rural.json" (which means simulation properties of adult females in the age range of 25-50 who are from rural areas), the transactions are created. Say, for this profile, you could check "Sparkov | Github | adults_2550_female_rural.json", there are parameter value ranges defined in terms of min, max transactions per day, distribution of transactions across days of the week and normal distribution properties (mean, standard deviation) for amounts in various categories. Using these measures of distributions, the transactions are generated using faker.

What I did was generate transactions across all profiles and then merged them together to create a more realistic representation of simulated transactions.

The app has currently been tailored for this dataset. This dataset has two parts, a training and a testing split. Both of these files are huge, hence i've chosen to take a 
small stratified subset of it which takes lesser time to train and can be trained on local hardware. 

If you're looking to adopt it to your dataset then it is important to look in the code found in strm.py to see if it fits your usecase.

This is the first iteration and was made as a part of college project work, further future improvements are on the way. 
