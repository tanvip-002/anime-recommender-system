# 🌸 Anime Recommender

A content-based anime recommendation app built with Python and Streamlit. Pick an anime you love and get up to 40 similar titles, shown five at a time with posters and match scores.

## 🔗 Live Demo

**[https://recmeanime.streamlit.app/](https://recmeanime.streamlit.app/)**

> The app runs on Streamlit Community Cloud's free tier, so it may be asleep after a period of inactivity. Click the wake-up button and it restarts in a few seconds.

---

## ✨ Features

- **Content-based recommendations** using genres, synopsis, studio and format
- **40 recommendations per search**, browsed five at a time with left and right arrows
- **Match score** on every card, with ranks that continue across pages
- **Automatic posters** fetched from AniList
- **Custom UI** with a dark navy and pink theme, frosted-glass panels and animated page transitions

## 🧠 How It Works

Similarity is computed offline in a notebook, and the app only loads the saved result.

1. **Cleaning:** missing English names fall back to the original name, and placeholder synopses and `Unknown` values are removed.
2. **Feature blocks:** each part of an anime is vectorized separately with TF-IDF.

   | Block | Details | Weight |
   |---|---|---|
   | Genres | Whole tokens, so `Slice of Life` and `Sci-Fi` stay intact | 0.4 |
   | Synopsis | Unigrams and bigrams, English stop words removed, 30,000 features | 0.4 |
   | Studios | Whole tokens | 0.1 |
   | Type | TV, Movie, OVA, etc. | 0.1 |

3. **Combining:** every block is L2-normalized, scaled by the square root of its weight, and stacked into one sparse matrix (`features.npz`). A dot product between two rows is a weighted average of the per-block cosine similarities.
4. **Serving:** the app scores the selected anime against all others, takes the top 40, and fetches their posters from the AniList GraphQL API in a single request.

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for the UI
- pandas, NumPy and SciPy (sparse matrices)
- scikit-learn (feature building in the notebook)
- [AniList GraphQL API](https://anilist.gitbook.io/anilist-apiv2-docs/) for posters, looked up by MyAnimeList ID

## 📁 Project Structure

```
anime-recommender-system/
├── app.py                # Streamlit app (UI + recommendation logic)
├── anime_meta.pkl        # MAL_ID and Name for each row of the feature matrix
├── features.npz          # Combined sparse feature matrix
├── requirements.txt      # Dependencies for deployment
└── .streamlit/
    └── config.toml       # Theme colors
```

`anime_meta.pkl` and `features.npz` must come from the same notebook run, since row `i` in one corresponds to row `i` in the other.

## 🚀 Run Locally

Clone or download this repository, then run:

```bash
cd anime-recommender-system

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 🔄 Rebuilding the Model

1. Load the MyAnimeList data (`anime.csv` and `anime_with_synopsis.csv`) and merge on `MAL_ID`.
2. Clean the data and turn `Genres`, `Studios` and `synopsis` into lists.
3. Build the TF-IDF blocks, apply the weights, and save:

```python
   sp.save_npz('features.npz', X)
   anime[['MAL_ID', 'Name']].to_pickle('anime_meta.pkl')
```

4. Commit both files together and push. Streamlit Cloud redeploys automatically.

## ☁️ Deployment

Hosted on Streamlit Community Cloud and deployed directly from this repository (`main` branch, `app.py`). Every push to `main` triggers a redeploy.

## ⚠️ Limitations

- Recommendations are content-based. They do not use user ratings or watch history.
- Sequels and side stories from the same franchise often rank highly because they share characters and plot text.
- Match percentages are a relative ranking, not a probability. Values of 30-70% are normal.
- Posters depend on AniList being reachable. If it is not, a placeholder image is shown.

## 🙏 Acknowledgements

- Anime metadata and synopses: MyAnimeList dataset (Kaggle)
- Poster images: [AniList](https://anilist.co/)

Posters and anime titles belong to their respective owners. This project is for learning and is not affiliated with MyAnimeList or AniList.
