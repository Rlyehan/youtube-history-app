import re
from datetime import datetime, timedelta
from typing import List, Union

import matplotlib.pyplot as plt
import nltk
import plotly.graph_objects as go
import polars as pl
import pytube
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def get_video_length(url) -> Union[int, None]:
    if url is None:
        return None

    try:
        yt = pytube.YouTube(url)
        return yt.length

    except Exception as e:
        st.error(e)
        return None


def get_video_keywords(url) -> Union[List[str], None]:
    if url is None:
        return None

    try:
        yt = pytube.YouTube(url)
        return yt.keywords

    except Exception as e:
        st.error(e)
        return None


def transform_data(df: pl.DataFrame) -> pl.DataFrame:
    # Split Video and Channel
    df = df.with_column(pl.col("subtitles").apply(lambda x: x[0]["name"]).alias(("channel")))
    # Convert time to datetime
    df = df.with_column(pl.col("time").str.strptime(pl.Datetime))
    df = df.with_column(pl.col('time').apply(lambda x: x.month).alias("month"))  # type: ignore
    df = df.with_column(pl.col("time").apply(lambda x: x.isocalendar().week).alias("week"))  # type: ignore
    df = df.with_column(pl.col("time").apply(lambda x: x.strftime('%A')).alias("day"))  # type: ignore
    # Keep only the columns we need
    df = df.select(["title", "channel", "titleUrl", "time", "month", "week", "day"])
    # Add video length & keywords
    with st.spinner("Fetching video data..."):
        df = df.with_columns([
            pl.col("titleUrl").apply(get_video_length).alias("video_length"),
            pl.col("titleUrl").apply(get_video_keywords).alias("keywords")
        ])
    return df


def calculate_different_timeframes(df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    # create different timeframe aggregations
    df_times = df.drop_nulls().sort("time")
    videos_per_day = df_times.groupby_dynamic("time", every="1d") \
        .agg([pl.count().alias("total videos"), pl.sum('video_length').alias('total length')]).sort("time")
    videos_per_channel = df.groupby("channel").agg(pl.count()).sort("count", reverse=True)
    videos_per_channel_per_month = df.groupby(["channel", "month"]).agg(pl.count()).sort("month", reverse=False)
    videos_per_channel_per_week = df.groupby(["channel", "week"]).agg(pl.count()).sort("count", reverse=True)
    average_videos_per_month = df.groupby("month").agg(pl.count()).mean()
    average_videos_per_week = df.groupby("week").agg(pl.count()).mean()
    average_videos_per_day = df_times.groupby_dynamic("time", every="1d").agg(pl.count()).mean()
    most_watched_channel = videos_per_channel["channel"][0]
    most_watched_channel_by_duration = df.groupby("channel").agg(pl.sum("video_length").alias("total length")) \
        .sort("total length", reverse=True)["channel"][0]
    most_watched_channel_per_month = videos_per_channel_per_month["channel"][0]
    most_watched_channel_per_week = videos_per_channel_per_week["channel"][0]

    return {
        "df_times": df_times,
        "videos_per_day": videos_per_day,
        "videos_per_channel": videos_per_channel,
        "videos_per_channel_per_month": videos_per_channel_per_month,
        "videos_per_channel_per_week": videos_per_channel_per_week,
        "average_videos_per_month": average_videos_per_month,
        "average_videos_per_week": average_videos_per_week,
        "average_videos_per_day": average_videos_per_day,
        "most_watched_channel": most_watched_channel,
        "most_watched_channel_by_duration": most_watched_channel_by_duration,
        "most_watched_channel_per_month": most_watched_channel_per_month,
        "most_watched_channel_per_week": most_watched_channel_per_week
    }


def plot_wordcloud(df: pl.DataFrame) -> None:
    df_keywords = df.with_columns([pl.col("keywords")]).explode("keywords").drop_nulls()
    keywords = " ".join(df_keywords["keywords"])

    tokens = []
    i = 0

    while i < len(df):
        tokens += word_tokenize(df_keywords['keywords'][i])
        i += 1

    keywords = [re.sub(r'[^\x00-\x7F]+', ' ', token) for token in tokens]

    stemmer = PorterStemmer()
    stemmed_keywords = [stemmer.stem(token) for token in keywords]

    stop_words = set(stopwords.words('english'))
    filtered_keywords = [token for token in stemmed_keywords if token not in stop_words and token.isalpha()]

    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_keywords = [lemmatizer.lemmatize(token) for token in filtered_keywords]

    keywords_string = " ".join(lemmatized_keywords)

    wordCloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(keywords_string)

    fig = plt.figure(figsize=(8, 8), facecolor=None, edgecolor=None)
    plt.imshow(wordCloud)
    plt.axis("off")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(
        page_title="Youtube watch History Analytics",
        page_icon=":guardsman:",
        layout="centered",
    )
    st.title("Youtube watch History Analytics")
    st.write("This is a simple app to analyze your youtube watch history")

    # File uploader
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")

    if uploaded_file is not None:
        try:
            df: pl.DataFrame = pl.read_json(uploaded_file)
        except Exception as e:
            st.error(e)
            return

        st.write("File uploaded successfully")

        # Transform data
        df = transform_data(df)

        # generic History stats
        st.write("## History Stats")
        st.write(f"Total number of videos: {df.shape[0]}")
        st.write(f"Total number of channels: {df['channel'].n_unique()}")
        st.write(f"Total number of unique videos: {df['title'].n_unique()}")
        total_time = df['video_length'].sum()
        total_time_minutes = round(total_time / 60, 2)
        total_time_hours = round(total_time / 3600, 2)
        st.write(f"Total time spent watching videos: {total_time_minutes} minutes or {total_time_hours} hours")

        min_date: datetime = min(df["time"])
        max_date: datetime = max(df["time"])
        date_diff: timedelta = max_date - min_date

        st.write(f"Your History spans from {min_date} to {max_date}, ({date_diff.days} days)")

        # create different timeframe aggregations
        timeframes = calculate_different_timeframes(df)

        # Display different timeframes
        st.write(f"Most watched channel by video count: {timeframes['most_watched_channel']}")
        st.write(f"Most watched channel by duration: {timeframes['most_watched_channel_by_duration']}")
        st.write(f"Most watched channel in a single month: {timeframes['most_watched_channel_per_month']}")
        st.write(f"Most watched channel in a single week: {timeframes['most_watched_channel_per_week']}")
        st.write(f"Average videos per month: {timeframes['average_videos_per_month']['count'][0]}")
        st.write(f"Average videos per week: {timeframes['average_videos_per_week']['count'][0]}")
        st.write(f"Average videos per day: {timeframes['average_videos_per_day']['count'][0]}")

        # Plot average duration per day
        videos_per_day = timeframes["videos_per_day"]
        avg_duration_watched_per_day = videos_per_day["total length"].mean()
        sec_perday = 86400

        pie_values = [avg_duration_watched_per_day, sec_perday - avg_duration_watched_per_day]
        pie_labels = ["Average duration watched per day", "Average duration not watched per day"]

        fig_avg_duration_day = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, pull=[0, 0.2])])
        st.plotly_chart(fig_avg_duration_day, use_container_width=False)

        # Plot duration watched per day
        fig_duration_day = go.Figure([go.Scatter(x=videos_per_day["time"], y=videos_per_day["total length"])])

        fig_duration_day.update_layout(
            title_text="Duration watched per day"
        )

        fig_duration_day.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                        ])
                    ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )

        st.plotly_chart(fig_duration_day, use_container_width=False)

        # Plot duration watched per month
        df_times = timeframes["df_times"]
        duration_watched_per_month = df_times.with_columns([
                                                pl.col("month"),
                                                pl.col("video_length").apply(lambda x: x / 3600).alias("video_length")
                                                ]).groupby("month").agg(pl.sum("video_length").alias("total length"))

        fig_duration_month = \
            go.Figure([go.Bar(x=duration_watched_per_month["month"], y=duration_watched_per_month["total length"])])
        fig_duration_month.update_layout(
            title_text="Duration watched per month"
        )
        st.plotly_chart(fig_duration_month, use_container_width=False)

        # Plot videos watched per channel
        videos_per_channel = timeframes["videos_per_channel"]
        fig_most_watched_channel = \
            go.Figure([go.Bar(x=videos_per_channel["channel"].head(10), y=videos_per_channel["count"].head(10))])
        fig_most_watched_channel.update_layout(
            title_text="Most watched channels"
        )
        st.plotly_chart(fig_most_watched_channel, use_container_width=False)

        # Plot wordcloud based on video titles
        plot_wordcloud(df)

    else:
        st.write("Upload a valid file to get started")


if __name__ == "__main__":
    main()