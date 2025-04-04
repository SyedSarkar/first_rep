import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the page config
st.set_page_config(page_title="Iris Data Dashboard", layout="centered")

st.title("🌸 Iris Dataset Analysis")

# Load dataset from the web
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Show the data table
st.subheader("📋 Raw Dataset")
st.dataframe(df)

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
species = st.sidebar.multiselect("Select species", df['species'].unique(), default=df['species'].unique())

filtered_df = df[df['species'].isin(species)]

# Show some basic stats
st.subheader("📊 Summary Statistics")
st.write(filtered_df.describe())

# Plot 1: Pairplot using seaborn
st.subheader("🔗 Seaborn Pairplot")
fig1 = sns.pairplot(filtered_df, hue="species")
st.pyplot(fig1)

# Plot 2: Boxplot
st.subheader("📦 Boxplot (Sepal Length by Species)")
fig2, ax = plt.subplots()
sns.boxplot(data=filtered_df, x="species", y="sepal_length", ax=ax)
st.pyplot(fig2)

# Plot 3: Correlation heatmap
st.subheader("🧠 Correlation Heatmap")
fig3, ax3 = plt.subplots()
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
st.pyplot(fig3)
