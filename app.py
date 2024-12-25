import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data['model']
le_country = data['le_country']
le_dev = data['le_dev']
le_education = data['le_education']

def show_predict_page():
    # Set background color and text color
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(to bottom, #001f3f, #001a33);
            color: white;
        }
        .stButton button {
            background-color: #004080;
            color: white;
        }
        h1, h2, h3, h4, h5, h6 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Salary Prediction")
    st.write("""### Please enter the required information to predict the salary""")
    
    countries = (
        'United States of America',
        'Germany',
        'United Kingdom',
        'Pakistan',
        'Canada',
        'France',
        'Brazil',
        'Spain',
        'Poland',
        'Australia',
        'Italy',
        'Sweden',
        'Russian Federation',
        'Switzerland',
        'Turkey',
        'Austria',
        'Portugal',
        'Norway',
        'Mexico'
    )
    
    dev_types = (
        'Developer, full-stack',
        'Developer, front-end',
        'Developer, back-end',
        'Data scientist or machine learning specialist',
        'Engineer, data',
        'Developer, mobile',
        'Developer, desktop or enterprise applications',
        'Engineer, site reliability',
        'Developer, embedded applications or devices',
        'Engineering manager',
        'DevOps specialist',
        'Developer, QA or test',
        'Academic researcher',
        'Data or business analyst',
        'Educator',
        'Senior Executive (C-Suite, VP, etc.)',
        'Developer, game or graphics',
        'Cloud infrastructure engineer'
    )
    
    education = (
        "Master's degree", 
        "Bachelor's degree", 
        'Less than a Bachelors',
        'Post grad'
    )
    
    country = st.selectbox("Country", countries)
    dev_type = st.selectbox("Type of job", dev_types)
    education = st.selectbox("Education Level", education)
    
    experience = st.slider("Years of experience", 0, 35, 3)
    
    ok = st.button("Calculate Salary")
    
    if ok:
        X = np.array([[country, dev_type, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_dev.transform(X[:,1])
        X[:, 2] = le_education.transform(X[:,2])
        X = X.astype(float)
        
        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
    
    st.write("""**Project by: Danish Ijaz Ahmad**""")

show_predict_page()
