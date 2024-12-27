import os
import base64
import streamlit as st
import pandas as pd
from tpot import TPOTRegressor, TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

vectorizer = TfidfVectorizer()
le = LabelEncoder()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'metric' not in st.session_state:
    st.session_state.metric = None
if 'best_pipeline' not in st.session_state:
    st.session_state.best_pipeline = None

# Title and Sidebar
st.title("Data Tool")
with st.sidebar:
    st.image("gojo_pic.jpeg")
    st.title("Options")
    choices = st.sidebar.radio("", ["Upload Dataset", "Perform EDA", "Select Task", "Get ML Pipeline", "Results"])

# Step 1: Upload Dataset
if choices == "Upload Dataset":
    st.header("Upload Dataset")
    file = st.file_uploader("Upload your file here, make sure its CSV!!")
    if file:
        st.session_state.df = pd.read_csv(file)
        st.session_state.df.to_csv("uploaded_dataset.csv", index=False)
        st.dataframe(st.session_state.df)

if os.path.exists("uploaded_dataset.csv") and st.session_state.df is None:
    st.session_state.df = pd.read_csv("uploaded_dataset.csv")

# Step 2: Perform EDA
if choices == "Perform EDA":
    st.title("Trying to do some EDA on the data")
    if st.session_state.df is not None:
        profile_report = ProfileReport(st.session_state.df)
        st_profile_report(profile_report)
    else:
        st.warning("Please upload your dataset first")

# Step 3: Select Task
if choices == "Select Task":
    if st.session_state.df is not None:
        st.header("Choose an Algorithm")
        st.session_state.task_type = st.selectbox(
            "Select the type of Machine Learning task:",
            ["Regression", "Classification", "Clustering"]
        )

        if st.session_state.task_type in ["Regression", "Classification"]:
            st.session_state.target_column = st.selectbox("Select the target column:", st.session_state.df.columns)
        st.success(f"Task '{st.session_state.task_type}' selected successfully!")
    else:
        st.warning("Please upload a dataset first!")

# Step 4: Run AutoML
if choices == "Get ML Pipeline":
    if st.session_state.df is not None:
        st.header("Generating ML Pipeline")

        if st.session_state.task_type == "Regression":
            target = st.session_state.target_column
            x = st.session_state.df.drop(columns=[target])
            y = st.session_state.df[target]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            st.write("Running TPOT Regressor...")
            with st.spinner("This might take a while..."):
                tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
                tpot.fit(x_train, y_train)

            st.session_state.best_pipeline = tpot.fitted_pipeline_
            y_pred = tpot.predict(x_test)
            st.session_state.metric = mean_squared_error(y_test, y_pred)
            # tpot.export("best_regressor_pipeline.py")
            st.success("TPOT Analysis is completed!")

        elif st.session_state.task_type == "Classification":
            target = st.session_state.target_column
            x = st.session_state.df.drop(columns=[target])
            y = st.session_state.df[target]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

            st.write("Running TPOT Classifier...")
            with st.spinner("This might take a while..."):
                tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
                tpot.fit(x_train, y_train)

            st.session_state.best_pipeline = tpot.fitted_pipeline_
            y_pred = tpot.predict(x_test)
            st.session_state.metric = accuracy_score(y_test, y_pred)
            tpot.export("best_classifier_pipeline.py")
            st.success("TPOT Analysis is completed!")

        elif st.session_state.task_type == "Clustering":
            st.write("Running KMeans Clustering...")
            x = StandardScaler().fit_transform(st.session_state.df)

            with st.spinner("This might take a while..."):
                kmeans = KMeans(n_clusters=3, random_state=42).fit(x)

            st.session_state.best_pipeline = kmeans
            st.session_state.metric = silhouette_score(x, kmeans.labels_)
            st.success("Clustering completed!")


# Step 5: View Results
if choices == "Results":
    if st.session_state.best_pipeline is not None:
        st.header("Results")
        st.write("Best Pipeline:")
        st.text(st.session_state.best_pipeline)
        st.write("Metric:", st.session_state.metric)

        if st.session_state.task_type in ["Regression", "Classification"]:
            file_name = "best_pipeline.py" if st.session_state.task_type == "Regression" else "best_classifier_pipeline.py"
            with open(file_name, "rb") as file:
                st.download_button(
                    label="Download Best Pipeline",
                    data=file,
                    file_name=file_name
                )
    else:
        st.warning("No results to display. Please run AutoML first!")
