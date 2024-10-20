import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

# Firebase setup
if not firebase_admin._apps:

    cred = credentials.Certificate("hacktheprompt-imagegen-firebase-adminsdk-rf4lq-e1569d8c0e.json")  # Replace with your Firebase credential file

    firebase_admin.initialize_app(cred, {

        'databaseURL': 'https://hacktheprompt-imagegen-default-rtdb.firebaseio.com/'

    })

# Fixed sequence of images and their corresponding actual prompts
image_prompts = [
    {"image": "image1.jpg", "prompt": "cute  Bird on a white background wool colorful"},
    {"image": "image2.jpg", "prompt": "drawing on a black background of a fish bowl"},
    {"image": "image3.jpg", "prompt": "close up of an chinchilla looking at the camera orange background"},
    {"image": "image4.jpeg", "prompt": "extreme closeup portrait of a high end mecha robot mirrored face shield metallic red and blue glowing eyes dark background"},
    {"image": "image5.jpeg", "prompt": "Robot Godzilla using chopsticks space rocket The rocket booster fires a yellow plume of fire at its bottom early morning on the beach ocean"}
]

# Function to calculate similarity between guessed and actual prompts
def calculate_similarity(guess, actual):
    vectorizer = TfidfVectorizer().fit_transform([guess, actual])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)
    return cosine_sim[0][1]  # Similarity score between 0 and 1

def score_LT(score):
  return 400+(score*599)

# Function to save score to Firebase
def save_scores_to_firebase(name,img_index, score):
    ref = db.reference('participants')
    ref.child(name).child(f'score{img_index}').set(score)

# Admin dashboard with password protection
def admin_view():
    st.title("Admin Dashboard")
    password = st.text_input("Enter admin password", type="password")
    if password == "aihacks.club@image-gen69":
        st.success("Access granted!")

        # Fetch and display participant scores from Firebase
        ref = db.reference('participants')
        data = ref.get()

        df=pd.DataFrame(columns=['Participant','Score1','Score2','Score3','Score4','Score5'])

        if data:
            for name, scores in data.items():
                row = [name,score_LT(scores['score1']),score_LT(scores['score2']),score_LT(scores['score3']),score_LT(scores['score4']),score_LT(scores['score5'])]
                df.loc[len(df)] = row





                # st.write(f"Participant: {name}")
                # st.write(f"Score 1: {scores['score1']}")
                # st.write(f"Score 2: {scores['score2']}")
                # st.write(f"Score 3: {scores['score3']}")
                # st.write(f"Score 4: {scores['score4']}")
                # st.write(f"Score 5: {scores['score5']}")
        else:
            st.write("No participants yet.")

        st.dataframe(df)
    else:
        st.error("Incorrect password. Access denied.")

# Participant view to submit guesses
def participant_view():
    st.title("Guess the Prompt")
    name = st.text_input("Enter your Team name")
    if name:
        for idx, img_info in enumerate(image_prompts):
            st.image(img_info["image"], caption=f"Describe Image {idx + 1}")
            guessed_prompt = st.text_input(f"Enter your prompt for Image {idx + 1}")
            if st.button(f"Submit for Image {idx + 1}"):
                actual_prompt = img_info["prompt"]
                score = calculate_similarity(guessed_prompt, actual_prompt)
                save_scores_to_firebase(name, idx + 1, score)
               # st.write(f"Your similarity score for Image {idx + 1} is: {score:.2f}")

    # save_scores_to_firebase(name, scores)



# Main application flow
mode = st.sidebar.selectbox("Select Mode", ("Participant", "Admin"))

if mode == "Participant":
    participant_view()
elif mode == "Admin":
    admin_view()
