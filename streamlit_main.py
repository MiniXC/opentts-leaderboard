import streamlit as st
import pandas as pd

df = pd.read_csv("results/data.csv")

st.dataframe(df, use_container_width=True)