import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page configuration for a premium feel
st.set_page_config(page_title="Bike Demand Clustering", page_icon="🚴‍♂️", layout="wide")

# Custom CSS for UI enhancements
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3, p, div {
        font-family: 'Inter', sans-serif;
    }
    /* Adding some subtle gradients to headers */
    h1 {
        background: -webkit-linear-gradient(#2EC4B6, #FF9F1C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255,255,255,0), rgba(255,255,255,0.75), rgba(255,255,255,0));
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏍️ Bike Demand Pattern Clustering Dashboard")
st.markdown("### Unlock insights from daily and hourly bike-sharing demand patterns using K-Means Clustering.")
st.markdown("---")

# ------------------------------------------
# LOAD DATA WITH CACHING
# ------------------------------------------
@st.cache_data
def load_data():
    hour = pd.read_csv("hour.csv")
    day = pd.read_csv("day.csv")
    return hour, day

try:
    hour_raw, day_raw = load_data()
except FileNotFoundError:
    st.error("Error: Could not find 'hour.csv' or 'day.csv' in the directory.")
    st.stop()

# ------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------
st.sidebar.header("⚙️ Configuration")
dataset_choice = st.sidebar.radio("Select Dataset to Analyze:", ["Hourly Data", "Daily Data"])
k_clusters = st.sidebar.slider("Number of Clusters (k):", min_value=2, max_value=8, value=3)

# Theme selection (user asked for preferences)
theme_choice = st.sidebar.selectbox("Select Color Theme", ["Vibrant", "Ocean", "Sunset", "Dark Magma"])

# Define custom palettes
color_palettes = {
    "Vibrant": ['#FF3366', '#2EC4B6', '#FF9F1C', '#8338EC', '#3A86FF', '#FF006E', '#FB5607', '#E0A96D'],
    "Ocean": ['#03045E', '#0077B6', '#00B4D8', '#90E0EF', '#48CAE4', '#ade8f4', '#caf0f8', '#023E8A'],
    "Sunset": ['#F94144', '#F3722C', '#F8961E', '#F9844A', '#F9C74F', '#90BE6D', '#43AA8B', '#577590'],
    "Dark Magma": ['#2E004B', '#5A0D65', '#8C1E70', '#C23267', '#EC5A53', '#FA9348', '#FFCE5B', '#FFF095']
}
selected_palette = color_palettes[theme_choice][:k_clusters]
if dataset_choice == "Hourly Data":
    df = hour_raw.copy()
    features = ['season', 'mnth', 'hr', 'holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
    x_axis_col = 'hr'
    x_axis_title = "Hour of the Day"
else:
    df = day_raw.copy()
    features = ['season', 'mnth', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'cnt']
    x_axis_col = 'weekday'
    x_axis_title = "Day of the Week"

# Select features
df_features = df[features]

# Normalize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Apply KMeans
kmeans = KMeans(n_clusters=k_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# ------------------------------------------
# DASHBOARD LAYOUT & VISUALS
# ------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 {dataset_choice} Clustering Visualization")
    
    # Configure Matplotlib Style
    plt.style.use("dark_background") # A modern dark theme for matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We create a mapping of colors for each cluster
    for cluster_id in range(k_clusters):
        cluster_data = df[df['Cluster'] == cluster_id]
        ax.scatter(
            cluster_data[x_axis_col], 
            cluster_data['cnt'], 
            c=selected_palette[cluster_id], 
            label=f'Cluster {cluster_id}',
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5
        )
        
    ax.set_xlabel(x_axis_title, fontsize=12, fontweight='bold', color='#E2E8F0')
    ax.set_ylabel("Bike Demand (Count)", fontsize=12, fontweight='bold', color='#E2E8F0')
    ax.set_title(f"{dataset_choice} vs Demand", fontsize=14, fontweight='bold', color='#E2E8F0')
    ax.legend(frameon=True, facecolor='#1E293B', edgecolor='none')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.tick_params(colors='#CBD5E1')
    ax.grid(True, linestyle='--', color='#334155', alpha=0.5)
    
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    
    st.pyplot(fig)

with col2:
    st.subheader("💡 Cluster Distribution")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    
    # Bar chart for distribution
    fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
    bars = ax_bar.bar(cluster_counts.index.astype(str), cluster_counts.values, color=selected_palette)
    
    ax_bar.set_xlabel("Cluster", color='#E2E8F0')
    ax_bar.set_ylabel("Number of Data Points", color='#E2E8F0')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['bottom'].set_color('#475569')
    ax_bar.spines['left'].set_color('#475569')
    ax_bar.tick_params(colors='#CBD5E1')
    ax_bar.grid(axis='y', linestyle='--', color='#334155', alpha=0.5)
    
    fig_bar.patch.set_facecolor('#0E1117')
    ax_bar.set_facecolor('#0E1117')
    
    st.pyplot(fig_bar)

st.markdown("---")
st.subheader("📈 Cluster Profile (Average Values)")
# Show average values of features for each cluster
cluster_means = df.groupby('Cluster')[features].mean().reset_index()

# Show a simple unstyled dataframe (Streamlit styles are better than Pandas styles currently in standard component without HTML conversion overhead)
st.dataframe(cluster_means, use_container_width=True)

st.markdown("---")
st.subheader("🔍 Sample Data Preview")
# Show a subset of the dataset
st.dataframe(df.head(15), use_container_width=True)
