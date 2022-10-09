
import streamlit as st
import numpy as np
import pickle
from PIL import Image
import base64
loaded_model=pickle.load(open('trained_model.sav', 'rb'))

tab1,tab2,tab3=st.tabs(["Home","Retinopathy Test","Contact"])
#
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(r'C:\Users\USER\Desktop\ExcelR Data Science 14122021\Project\ExcelR\Project_1\final_ppt\Files\Model_Deploy_07102022\157601-cataract-surgery-and-diabetic-retinopathy.jpg')
#

with tab1:
    st.header("Home")
    st.markdown("Diabetic retinopathy is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). At first, diabetic retinopathy might cause no symptoms or only mild vision problems")
    
    
    
with tab2:
    st.header("Retinopathy Test")
    def retinopathy_prediction(input_data):

        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        prediction = loaded_model.predict(input_data_reshaped)
        print(prediction)

        if (prediction[0] == 0):
          return "The person doesn't has retinopathy"
        else:
          return "The person has retinopathy"

    def main():
        
    
        #Getting the required data
        age = st.text_input("Enter your Age")
        systolic_bp = st.text_input("Enter your Systolic_bp Level")
        diastolic_bp = st.text_input("Enter your Diastolic_bp Level")
        cholesterol = st.text_input("Enter your Cholesterol Level")

        #code for prediction
        prognosis= ''

        #creating prediction button

        if st.button("Retinopathy Result"):
          prognosis =  retinopathy_prediction([age , systolic_bp , diastolic_bp , cholesterol])

        st.success(prognosis)
        
    if __name__ == "__main__":
      main()
    

with tab3:
    st.header("Contact")
    st.subheader("Raj Jadhav   email:rajjadhav@gmail.com")
    st.subheader("Ujjwal Bhole email:ubhole9@gmail.com")
    st.subheader("Sovan Nandi  email:sovaneee@gmail.com")
    st.subheader("Kaushal Kalantri   email:kaushalkalantricareer@gmail.com")  
    st.subheader("Sanjeet Dhume  email:sanjeetdhume97@gmail.com")
    st.subheader("anitha  email:anitaani50000@gmail.com")





