"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import string
from pathlib import Path

# Function to import markdowns
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

# Function to clean text
def data_refine(text):
	# convert to lower case
	text = text.lower()
	# remove punctuations
	text = text.translate(str.maketrans('', '', string.punctuation))
	# remove hashtags and @ signs
	new_text = text.replace('#','').replace('@','')

	return new_text

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Climate Man :earth_africa: ")
	st.subheader("A Climate change tweet classifier :partly_sunny_rain:")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Homepage", "Information", "Prediction", "About us"]
	selection = st.sidebar.selectbox("What do you need?", options)

	if selection == "Homepage":
		st.image('resources/imgs/climate_man_3.jpg', width= 400)
		st.subheader("Welcome to ``Climate Man``. The app you need for all your Climate change tweet classifications")
		st.subheader("brought to you by ``DataTech``:registered:")
		st.write("Hint: Navigate through the app with the side bar") 

	# Building out the "Information" page
	if selection == "Information":
		options = ["make a choice here", "General Information", "EDA", "Model Information"]
		selection = st.sidebar.selectbox("What do you want to know?", options)

		if selection == "make a choice here":
			st.image('resources/imgs/info_page.png', width= 500)
			st.subheader("What kind of info do you need :question:")
			st.subheader(":arrow_upper_left: Make your choice from the side bar")

		if selection == "General Information":
			st.info("General Information")
			# You can read a markdown file from supporting resources folder
			info_markdown = read_markdown_file("resources/info.md")
			st.markdown(info_markdown)

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page

		if selection == "EDA":
			st.info("Exploratory Data Analysis")
			# Buiding out the EDA page
			st.subheader("Here are some awesome visuals that show some of the insights we got from the data")
			st.image('resources/imgs/sentiment_bar.png', width= 500)
			st.write("This shows that over 8000 tweets were `Pro` tweets.")
			st.write("A lot of people support Climate change!!")
			st.subheader("   ")# just a way to create space between texts
			st.subheader("   ")

			st.image('resources/imgs/sentiment_pie.png', width= 500)
			st.write("This pie chart confirms the claim we had.")
			st.write("More than half(`over 50%`) of consumers believe in Climate change!!")
			st.subheader("   ")# just a way to create space between texts
			st.subheader("   ")

			st.image('resources/imgs/wordcloud.png', width= 800)
			st.write("This word-cloud data shows the most occuring words in each sentiment class.")
			st.write("With this we can see that our data source is accurate :sunglasses:")
			st.subheader("   ")# just a way to create space between texts
			st.subheader("   ")
		
		if selection == "Model Information":
			st.info("Model Information")
			# You can read a markdown file from supporting resources folder
			info_markdown = read_markdown_file("resources/models.md")
			st.markdown(info_markdown)

	# Building out the predication page
	if selection == "Prediction":
		options = ["make a choice here", "Logistic Regression", "XGBoost", "Naive Bayes", "Random Forest", "K-Nearest Neighbours"]
		selection = st.sidebar.selectbox("Choose a Model", options)

		if selection == "make a choice here":
			st.image('resources/imgs/model.jpg', width= 500)
			st.subheader("Which of the Models would you like to use :question:")
			st.subheader(":arrow_upper_left: Make your choice from the side bar")
		
		if selection == "Logistic Regression":
			st.info("Prediction with Logistic Regression")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_refine(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:bookmark_tabs:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:heavy_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:snowflake:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))
	

		if selection == "XGBoost":
			st.info("Prediction with XGBoost")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_refine(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/XGBoost.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:bookmark_tabs:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:heavy_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:snowflake:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))
	
		
		if selection == "Naive Bayes":
			st.info("Prediction with Naive Bayes")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_refine(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Naive_bayes.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:bookmark_tabs:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:heavy_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:snowflake:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))
	
		
		if selection == "Random Forest":
			st.info("Prediction with Random Forest")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_refine(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Random_forest.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:bookmark_tabs:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:heavy_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:snowflake:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))
	
		
		if selection == "K-Nearest Neighbours":
			st.info("Prediction with K-Nearest Neighbours")
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# cleaning user input
				clean_text = data_refine(tweet_text)

				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([clean_text]).toarray()

				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/Knn.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				if prediction == [2]:
					st.success("Text Categorized as: {}  :left_right_arrow:  News:bookmark_tabs:".format(prediction))
				
				if prediction == [1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Pro:heavy_check_mark:".format(prediction))
				
				if prediction == [0]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Neutral:snowflake:".format(prediction))
				
				if prediction == [-1]:
					st.success("Text Categorized as: {}  :left_right_arrow:  Anti:x:".format(prediction))

	
	if selection == "About us":
			st.info("About Us")
			# Building out the 'about us' page
			st.image('resources/imgs/the_team.jpeg', width= 700)
			st.subheader("`Mission Statement:`:page_with_curl::page_with_curl:")
			st.subheader("At DataTech:registered:, we assist businesses in making informed decisions\
						through application of data science techniques in solving real world problems.")
			st.subheader("We pride ourselves in applying the latest technology to provide actionable intel,\
						in turn helping businesses to grow and nurture a consumer-first mindset.")
			st.subheader("   ")# just a way to create space between texts
			st.subheader("   ")
			st.subheader("`Meet The Team:`:male-technologist::female-factory-worker::man-raising-hand:")

			st.subheader("   ")
			# First Member
			st.image('resources/imgs/obeng.jpg', width= 300)
			st.text('Emmanuel Obeng Afari')
			st.write('`Data Scientist`', '`Team Lead`')

			st.subheader("   ")
			# Second Member
			st.image('resources/imgs/kenny.jpeg', width= 300)
			st.text('Kenny Ozojie')
			st.write('`Data Scientist`', '`Product Lead`')

			st.subheader("   ")# just a way to create space between texts
			# Third Member
			st.image('resources/imgs/maryam.jpg', width= 300)
			st.text('Maryam Quadri')
			st.write('`Data Scientist`', '`Admin Lead`')

			st.subheader("   ")
			# Fourth Member
			st.image('resources/imgs/ndi.jpeg', width= 300)
			st.text('Ndinannyi Mukwevho')
			st.write('`Data Scientist`', '`Tech Lead`')

			st.subheader("   ")# just a way to create space between texts
			# Fifth Member
			st.image('resources/imgs/jide.jpg', width= 300)
			st.text('Babajide Adelekan')
			st.write('`Data Scientist`', '`PR Lead`')

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
