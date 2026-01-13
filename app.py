import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.spatial import distance

st.set_page_config(page_title="Spotify UMAP Recommender", layout="wide")

@st.cache_data
def load_data() -> pd.DataFrame:
    track_data = pd.read_pickle("data/techno_umap_embeddings.pkl")
    
    return track_data

def main():
    track_data = load_data()
    
    st.title("🌌 Galaxy Explorer")
    selected_genres = st.multiselect("Select Genres", options=track_data['track_genre'].unique(), default=track_data['track_genre'].unique())
    pop_range = st.slider("Minimum Popularity", 0, 100, 20)
    
    filtered_track_data = track_data.loc[(track_data['track_genre'].isin(selected_genres)) & (track_data['track_popularity'] >= pop_range)]
    
    tab1, tab2, tab3 = st.tabs(["🌌 Galaxy Explorer", "🔍 Search & Recommend", "Raw List"])
    
    with tab1:
        st.subheader("Interactive Music Universe")
        fig = px.scatter(
            filtered_track_data, x='x', y='y', 
            color='track_genre', 
            hover_data=['track_artist', 'track_name'],
            template='plotly_dark',
            height=600, 
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
    
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.subheader("Find Recommendations")
    
        search_query = st.text_input("Search for a Song Name", "")
        if filtered_track_data.empty:
            st.error("No tracks available with the selected filters. Please adjust your genre selection or popularity range.")
            return
        if search_query:
            matches = filtered_track_data[filtered_track_data['track_name'].str.contains(search_query, case=False, na=False)]
            selected_song_name = st.selectbox("Select the exact track:", 
                                              matches['track_name'],
                                              format_func=lambda x: f"{x} by {matches[matches['track_name'] == x]['track_artist'].values[0]}")
            selected_song = filtered_track_data[filtered_track_data['track_name'] == selected_song_name].iloc[0]
            
            if not selected_song.empty:
                target_coords = (selected_song['x'], selected_song['y'])

                filtered_track_data['distance'] = filtered_track_data.apply(lambda row: distance.euclidean(target_coords, (row['x'], row['y'])), axis=1)
                max_distance = filtered_track_data["distance"].max()
                filtered_track_data["relevance"] = 1 - (filtered_track_data["distance"] / max_distance)

                number_of_recommendations = min(101, len(filtered_track_data) - 1)
                number_of_recommendations_select = st.selectbox("Number of Recommendations", options=list(range(5, number_of_recommendations, 5)))
                recommendations = filtered_track_data[filtered_track_data['track_id'] != selected_song['track_id']].sort_values('distance').head(number_of_recommendations_select)
                st.write(f"### Top 20 Recommendations for '{selected_song_name} by {selected_song['track_artist']}'")
                
                column_config = {
                    "track_name": st.column_config.TextColumn(
                        "Name"
                    ),
                    "track_artist": st.column_config.TextColumn(
                        "Artist"
                    ),
                    "track_genre": st.column_config.TextColumn(
                        "Genre"
                    ),
                    "track_popularity": st.column_config.NumberColumn(
                        "Popularity",
                        format="%d"
                    ),
                    "distance": st.column_config.NumberColumn(
                        "Euclidean Distance",
                        help="Lower distance indicates higher similarity",
                        format="%.5f"
                    ),
                    "relevance": st.column_config.ProgressColumn(
                        "Relevance",
                        help="Higher relevance indicates higher similarity",
                        min_value=0,
                        max_value=1
                    )
                }

            st.dataframe(recommendations[['track_name', 'track_artist', 'track_genre', 'track_popularity', 'distance', "relevance"]],width='stretch', hide_index=True, column_config=column_config)

        else:
            st.warning("No songs found with that name.")
            
    with tab3:
        st.subheader("Raw Track List")
        st.dataframe(filtered_track_data.reset_index(drop=True))
        
    
if __name__ == "__main__":
    main()