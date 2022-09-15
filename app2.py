
#### IMPORT LIBRARIES ####
import streamlit as st
import os
import pickle
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dashboard", layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

####### LOAD PICKLE FILES #######
with open("pickle_files\gbc_best.pkl", "rb") as f: # gbc model
    gbc_best = pickle.load(f)

with open("pickle_files\\rfc_model.pkl", "rb") as f: # svc model
    rfc_model = pickle.load(f)

with open("pickle_files\\adb_model.pkl", "rb") as f: # gnb model
    adb_model = pickle.load(f)

with open("pickle_files\ohe_columns.pkl", "rb") as f: # one hot encoded columns.
    ohe_df_columns = pickle.load(f)

with open("pickle_files\imputer.pkl", "rb") as f: # imputer
    imputer = pickle.load(f)

with open("pickle_files\scaler.pkl", "rb") as f: # scaler
    scaler = pickle.load(f)

with open("pickle_files\label_encoder.pkl", "rb") as f: # label encoder
    label_en = pickle.load(f)

with open("pickle_files\syn_data_model.pkl", "rb") as f:
    syn_model = pickle.load(f)

##### FUNCTIONS #####
def get_prediction(df_row):
    input_df_ohe = pd.get_dummies(df_row).reindex(columns = ohe_df_columns, fill_value=0)
    input_df_ohe = pd.DataFrame(imputer.transform(input_df_ohe), columns=input_df_ohe.columns)
    input_df_ohe = scaler.transform(input_df_ohe)

    gbc_pred = gbc_best.predict(input_df_ohe)
    rfc_pred = rfc_model.predict(input_df_ohe)
    adb_pred = adb_model.predict(input_df_ohe)
    
    combined_pred = [gbc_pred[0], rfc_pred[0], adb_pred[0]]
    if combined_pred.count(1) >= 2:
        pred = 1
    else:
        pred = gbc_pred[0]
    pred = label_en.inverse_transform([pred])
    return pred

def generate_data(amount):
    syn_data = syn_model.sample(amount)
    syn_data["Churn"] = label_en.inverse_transform(syn_data["Churn"])
    with open("syn_data.pkl", "wb") as f:
        pickle.dump(syn_data, f)

def plotly_age(syn_data):
    fig = px.histogram(data_frame = syn_data,
                x = 'Ages',
                color = 'Churn',
                nbins = 91
                )
    fig.update_layout({"title": f'Age Histogram by Churn',
                        "xaxis": {"title":"Age"},
                        "yaxis": {"title":"Count"},
                        "showlegend": True})
    return fig

def plotly_completed(syn_data):
    fig = px.histogram(data_frame = syn_data,
                x = 'Completed',
                color = 'Churn',
                nbins = 100
                )
    fig.update_layout({"title": f'Completed Hours Histogram by Churn',
                        "xaxis": {"title":"Completed Hours"},
                        "yaxis": {"title":"Count"},
                        "showlegend": True})
    return fig

def plot_donuts(syn_data, col_name):
    df = syn_data
    vals = list(round(df[col_name].sort_index().value_counts(normalize=True)*100, 2).values)
    labels = list(round(df[col_name].sort_index().value_counts(normalize=True)*100, 2).index)

    fig = go.Figure(data = go.Pie(values = vals, 
                                labels = labels, hole = 0.5))
    fig.update_traces(hoverinfo='label+percent',
                    textinfo='percent', textfont_size=20)
    fig.update_layout(
                    title_text = f'{col_name} Donut Chart',
                    title_font = dict(size=25,family='Verdana'))
    fig.add_annotation(x= 0.5, y = 0.5,
                        text = f'Observations:\n{len(df)}',
                        font = dict(size=13,family='Verdana', 
                                    color='cyan'),
                        showarrow = False)
    return fig

def get_reports(syn_data):
    y_true = label_en.transform(syn_data["Churn"])
    input_df_ohe = pd.get_dummies(syn_data).reindex(columns = ohe_df_columns, fill_value=0)
    input_df_ohe = pd.DataFrame(imputer.transform(input_df_ohe), columns=input_df_ohe.columns)
    input_df_ohe = scaler.transform(input_df_ohe)
    gbc_pred = gbc_best.predict(input_df_ohe)
    report = classification_report(y_true, gbc_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    fig = plt.figure()
    sns.heatmap(confusion_matrix(y_true, gbc_pred), annot=True, fmt=".0f", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    return report_df, fig

#### SIDE BAR INSTANTIATED ####
with st.sidebar:
    title = "Dashboard"
    st.title(title)
    st.write("Welcome to the Churn Prediction Dashboard!")

#### EXPANDER FOR FORM ####
with st.expander("Churn Prediction Form"):
    #### FORM INPUTS ####
    with st.form("churn_form"): 
        st.write("Before embarking on your Data Science journey,")
        expertise_input = st.select_slider("Level of Expertise in Programming:", ("Beginner", "Intermediate", "Advanced"), value="Intermediate")
        age_input = st.slider("Age:", 20, 65, step=1, value=24)
        gender_input = st.radio("Gender:", ("Male", "Female"))
        emp_input = st.selectbox("Employment status:", ("Unemployed", "Employed", "Self-Employed", "Other (Retired, etc.)"))
        parti_input = st.select_slider("Participation level (classes, labwork, etc.):", ("Low", "Medium", "High"), value="Medium")
        level1_input = st.number_input("Level1", 1, 333, step=1)
        level2_input = st.number_input("Level2", 0, 333, step=1)
        level3_input = st.number_input("Level3", 0, 333, step=1)
        hrs_input = st.select_slider("Hours of study per week:", ("0-10", "10-20", "20-30", "30-40"))
        time_study_input = st.select_slider("General Study Timing:", ("00:00-06:00", "06:00-12:00", "12:00-18:00", "18:00-00:00"))
        time_counts_input = st.slider("Time Counts", 1, 4, step=1)
        state_input = st.radio("State", ("Active", "Inactive"))

        submitted = st.form_submit_button("Estimate Churn Likelihood")
        if submitted:
            completed_input = level1_input + level2_input + level3_input
            if expertise_input == "Beginner":
                expertise_input = "beginner"
            parti_input = parti_input.lower()
            col_names = ['Ages', 'Level1', 'Level2', 'Level3', 'Completed', 'Time counts',
        'Gender', 'Expertise', 'Commitment', 'Time', 'State']
            input_data = [age_input, level1_input, level2_input, level3_input, completed_input, time_counts_input, gender_input, expertise_input, parti_input, time_study_input, state_input]
            input_df = pd.DataFrame(columns=col_names)
            input_df.loc[0] = input_data
            pred = get_prediction(input_df)
            with st.spinner("Estimating....."):
                time.sleep(1.5)
            st.info("Likelihood to Churn: " + pred[0].title())

#### SIDE BAR FUNCTIONS ####
gen_input = st.sidebar.number_input("Generate Synthetic Data", 5000, 50000, 10000, 5000)
st.sidebar.write('The current number of rows is ', gen_input)
gen_data = st.sidebar.button("Generate")
if gen_data:
    with st.spinner("Generating Synthetic Data...(ETA 2 Seconds)"):
        time.sleep(1.5)
        syn_data = generate_data(gen_input)
        st.sidebar.success(f"{gen_input} rows of data generated!")

view_syn_dataset = st.sidebar.button("Display Generated Dataset")
if view_syn_dataset:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        st.write(syn_data)
        st.success("Evaluation Successful!")

eval_data = st.sidebar.button("Evaluate Churn with Pre-Trained Model")
if eval_data:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        with st.spinner("Evaluating....(ETA: 5 Seconds)"):
            clf_report, cmx = get_reports(syn_data)
        st.write(clf_report, use_container_width = True)
        st.pyplot(cmx)


#### SYNTHETIC DATA ANALYTICS ####
st.sidebar.subheader("-- Synthetic Data Analytics --")
plot_age_data = st.sidebar.button("Plot Age Distribution")
if plot_age_data:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plotly_age(syn_data)
        st.plotly_chart(fig, use_container_width = True)

plot_completed_data = st.sidebar.button("Plot Completed Hours Distribution")
if plot_completed_data:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plotly_completed(syn_data)
        st.plotly_chart(fig, use_container_width = True)

plot_donuts_button = st.sidebar.button("Plot Expertise")
if plot_donuts_button:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plot_donuts(syn_data, "Expertise")
        st.plotly_chart(fig, use_container_width = True)

plot_gender_button = st.sidebar.button("Plot Gender")
if plot_gender_button:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plot_donuts(syn_data, "Gender")
        st.plotly_chart(fig, use_container_width = True)


plot_Commitment_button = st.sidebar.button("Plot Commitment")
if plot_Commitment_button:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plot_donuts(syn_data, "Commitment")
        st.plotly_chart(fig, use_container_width = True)

plot_time_button = st.sidebar.button("Plot Time Intervals")
if plot_time_button:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plot_donuts(syn_data, "Time")
        st.plotly_chart(fig, use_container_width = True)

plot_state_button = st.sidebar.button("Plot Status Ratio")
if plot_state_button:
    if not os.path.exists("syn_data.pkl"):
        st.sidebar.error("Generate the data first!")
    else:
        with open("syn_data.pkl", "rb") as f:
            syn_data = pickle.load(f)
        fig = plot_donuts(syn_data, "State")
        st.plotly_chart(fig, use_container_width = True)
