
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gensim.models.keyedvectors as word2vec
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


page_bg_img = f"""                  
<style>               
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://miro.medium.com/v2/resize:fit:920/0*RTGZBvYgQQyyTFXZ.jpg");
 
background-position: top;
 
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)     # used to apply this styling to the Streamlit application

st.markdown(
    """
    <style>
    .st-emotion-cache-13ln4jf {
        max-width: none !important;
        padding: 10px;
    }
     .st-emotion-cache-165ax5l{
      background-color: rgba(0, 0, 0, 0.5);
      border-radius: 20px;
      color:white;

     }
     .st-emotion-cache-10trblm{
      color: floralwhite;
    font-palette: light;
    text-align: center;
     background-color: rgba(0, 0, 0, 0.5);
      border-radius: 20px;

     }

     .st-emotion-cache-a51556{
      color:white;
      font-size: larger;
      font-weight: bolder;
     }
    """,
    unsafe_allow_html=True
)
# Create a function for tokenizer

stemmer = nltk.stem.PorterStemmer()
ENGLISH_STOP_WORDS = stopwords.words('english')

@st.cache_data(show_spinner=False)
def recipe_tokenizer(sentence):
    # remove punctuation and set to lower case
    for punctuation_mark in string.punctuation:
        sentence = sentence.replace(punctuation_mark,'').lower()

    # split sentence into words
    listofwords = sentence.split(' ')
    listofstemmed_words = []

    # remove stopwords and any tokens that are just empty strings
    for word in listofwords:
        if (not word in ENGLISH_STOP_WORDS) and (word!=''):
            # Stem words
            stemmed_word = stemmer.stem(word)
            listofstemmed_words.append(stemmed_word)

    return listofstemmed_words

# Function to load the combined embeddings and TF-IDF vectorizer model
@st.cache_data(show_spinner=False)
def load_embeddings_and_vectorizer():
    with open('combined_embeddings.pkl', 'rb') as f:
        combined_embeddings = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return combined_embeddings, vectorizer

# Function for finding recipes
@st.cache_data(show_spinner=False)
def find_similar_recipes(sampled_data, user_input, num_similar=5):
    combined_embeddings, vectorizer = load_embeddings_and_vectorizer()

    # Process user input
    # Create a DataFrame for user input
    user_data = pd.DataFrame({'text_data': [user_input]})
    user_data['text_data'] = user_data['text_data'].str.lower()

    # Vectorize the user input using the provided vectorizer
    user_vectorized_data = vectorizer.transform(user_data['text_data'])

    # Ensure the number of features in user_vectorized_data matches with combined_embeddings
    num_missing_features = combined_embeddings.shape[1] - user_vectorized_data.shape[1]
    if num_missing_features > 0:
        # Add zero columns to user_vectorized_data to match the number of features
        user_vectorized_data = np.pad(user_vectorized_data.toarray(), ((0, 0), (0, num_missing_features)))

    # Compute cosine similarity with user input
    cosine_sim_matrix = cosine_similarity(user_vectorized_data, combined_embeddings)

    # Retrieve similar recipe indices
    similar_recipes = cosine_sim_matrix[0].argsort()[::-1][:num_similar]

    # Get similar recipe names from food_df
    similar_recipe_names = sampled_data.iloc[similar_recipes]['name'].tolist()

    return similar_recipe_names

# Function for loading data
@st.cache_data(show_spinner=False)
def load_data(url):
    df = pd.read_pickle(url)
    return df

data = load_data("food.pkl")
sampled_data = data.sample(frac=0.25, random_state=42)

# Define the app title and description
st.title('Food Recommendation!!') 
st.write("""
    <div style="color: #ffffff; font-size: 18px; text-align: center; padding: 20px; background-color: rgba(0, 0, 0, 0.5);border-radius: 20px;text-align: justify;font-size: 18px;font-weight: 700;">
        Discover exciting new recipes tailored to your cravings and ingredients! Whether you're seeking inspiration or a delightful surprise, this ML-powered app will scour through 20,000+ recipes to find your perfect match. Don't let those ingredients go to waste — unleash your culinary creativity now!  
    </div>
""", unsafe_allow_html=True)


st.divider()

# User input
user_input = st.text_input("", "", placeholder="Enter the craving here!!")


# Function to convert list of ingredients to comma-separated string
def format_ingredients(ingredients_list):
    # Remove the brackets and single quotes
    return ingredients_list.replace("[", "").replace("]", "").replace("'", "")

def format_description(text:str):
    info = text.replace("[", "").replace("]", "").replace("'", "")
    cap = info.capitalize()
    return (cap[:77] + '..') if len(cap) > 77 else cap

def format_name(text:str):
    return text.capitalize()

def style_table(df):
    styles = [
        dict(selector="th", props=[("color", "white"), ("font-weight", "bold"), ("background-color", "black"), ("text-align", "center"), ("word-wrap", "break-word"), ("width", "100%")]),
        dict(selector="td", props=[("color", "white"), ("background-color", "rgba(0, 0, 0, 0.5)"), ("text-align", "center"), ("word-wrap", "break-word"), ("width", "100%")]),
        dict(selector=".row0", props=[("background-color", "rgba(0, 0, 0, 0.5)")])
    ]
    return df.style.set_table_styles(styles)
if st.button('Get Recommendations!'):
    
    # Add loading screen
   # loading_screen = st.image('/content/drive/MyDrive/Recipe_project/files/input/cooking.gif', caption="Picking out deliciousness...")
    
    # Get recommendations based on user input
    recommended_recipes = find_similar_recipes(sampled_data, user_input, num_similar=5)
    
    # Remove loading screen
     
    
    # Get the selected recipes from the sampled_data DataFrame
    selected_recipes = sampled_data.loc[sampled_data['name'].isin(recommended_recipes), ['name', 'description', 'ingredients']]

    # Reset the index of selected_recipes to start from 1
    selected_recipes.reset_index(drop=True, inplace=True)
    selected_recipes.index += 1
    
    # Convert the list of ingredients into a comma-separated string
    selected_recipes['name'] = selected_recipes['name'].apply(format_name)
    selected_recipes['ingredients'] = selected_recipes['ingredients'].apply(format_ingredients)
    selected_recipes['description'] = selected_recipes['description'].apply(format_description)

    # Display the similar recipes in a table
    st.table(selected_recipes)
