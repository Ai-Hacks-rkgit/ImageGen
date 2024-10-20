import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

# Firebase setup
if not firebase_admin._apps:

    cred = credentials.Certificate('''{
  "type": "service_account",
  "project_id": "hacktheprompt-imagegen",
  "private_key_id": "e1569d8c0ef94160d119c66af5f4a7a2146823db",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDMjLzXnoFwVEh8\nXrvKzuJTjmGODponC7okLvxGowZToq59jf8Y1S7fTAIBblEuYjCDhwmElqxLDtfP\nI9Be3DC9/t0/zQdSq4SSaU0I+hCDbi73tig3WjoFrceTPxNdLJ2UzPDqzicKBpSv\nqt0Zd7p3tpl9ypq/COF7Upbt5XioYH1CHF1To062oGIJjSegeklBsNpV7an0CBWn\n5rAfn8xOpqBN4k2v0NwnVcZzQ1hwC1HV+Z3k5f7udROX6++Y8rX9uM0ZQB4yNjfQ\nQXv8UxNyukhU5MW3jWLcnimSvIRZdiyP/2yascpjCP7ITUQ2e72iHV/mn1RRnyQC\nxUS8i/4BAgMBAAECggEABz3WSQCSDuLU/B4zIh9ykMhiI3At50L5heK/SKrohEPR\nT0hHJzRo5+aIzaR/rZeJUHwSi1tK4eS2dt1gOPOCVZT1H+ycx+xNkbdg+aKMvMfE\n/ue32G8nzbVUyEg74cO8wnICTMsb7mAZmcx/7lu4QhkN/w+SAesJEd90e3CbyxjV\nz9fMGdFyrZGtW9mxlS9NPgYN24H8ODXhb+YXvTpQCrMXsyDtSxQWTQUFh433JWOT\n4zzZjtbgoTx0Fo9S/E2z9n6z/0y+suN5Xt/v1E1GMPIHe5hW8zrt+Mncv401ccQd\nEijH73CgczRq+7mYy92Z/25EAzEDkOnEPjviXtFFFwKBgQD369BB6lEpmnxeGOKm\n6Mieu9AIMz5sQBCE2alJMQfxwcnFWOdbsKNzpQ25ht6R9pw3VYNQsltsNdGeI0P9\nk6TdlfDfxG+OuuiFgUOw8BP3wjebsR9OaPsj9JXknJCJMdUG5jl0LycETwkrVzLO\nNcYDk66YJ8X9w4hhiO5Q9vChTwKBgQDTNx11iz58rcO8a5zBPcQeahPPQoi6DKq3\n1YTPxZjosL6BYDIDFHxP++Rt4fxbJqgZSkKvuc2WeoJ4FXtT1rPlXdUTkA6pb4Yp\nEO4Cx0ZH5FgvJ5HqU0+GWAYv87GnmIQgy/BduMVkpp1nL3/Nk9+wMjCJwmB47tq8\nz9oDA053rwKBgQCyrrDAcSLh+0fbgdAJQAkn7nD3GAfLeTjupvmNmNsC8Qp9Q6Ar\nw1lqxfDoYD4VsUnRz73+8S1XBkr30K72Ge1fDuw2Opu0oR1o60tgQQgDL2VovvWz\nS5KFzYgi5nx9hP7mJBQQmtNiFZykMgqZ+MOoXE4ft7rJNJ4cvdYVYIT5nQKBgQCf\nePurJkk1xdUFzJJ8bPBIrnrqgCfPoYS8bGBsp5q+BcSw1jqsjKkXku5z8K6i+9rr\nzV/wYe9R8InVtRJ6yJ7nTSN2M8x+LZA0LW4nduIfoc7bO5s2O1TN8GQrjGnUSplo\nUdLYUIvpZMtvfzOVulKoLBztxm8kn+NTr/PBVpvGTQKBgD5GLKGmZ076qLEr0dR5\nkUnuWTONkkyUPY3C5t8ZDcWfE+rxv0Gn6HQ+nDubluU1PQ5cN95c7Y2GvdMaaw1Z\nn33vE6c1Zar3MUPBibGPSQ9WI7TFx1cDswNB38LrBECe+4bBoCLewVuvkec8gGAZ\nO0Aed4fjkJPHEIQc5QLpcJbh\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-rf4lq@hacktheprompt-imagegen.iam.gserviceaccount.com",
  "client_id": "115325019680223120359",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-rf4lq%40hacktheprompt-imagegen.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}''')

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
