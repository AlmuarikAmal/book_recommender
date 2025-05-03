import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())

def retrieve_semantic_recommendations(query:str,
                                     category:str = None,
                                     tone:str = None,
                                     initial_top_k=50,
                                     final_top_k= 16) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books.loc[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs.loc[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by= "joy", ascending=False, inplace=True)
    if tone == "Surprising":
            book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
            book_recs.sort_values(by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
            book_recs.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
            book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
            query:str,
            category:str,
            tone:str,
    ):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        title = row["title"]
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_desc = " ".join(truncated_desc_split[:30]) + "..."

        author_split = row["authors"].split(";")
        if len(author_split) == 2:
            authors_str = f"{author_split[0]} and {author_split[1]}"
        elif len(author_split) > 2:
            authors_str = f"{', '.join(author_split[:-1]) + ", and " + author_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{title} by {authors_str} | {truncated_desc}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme= gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgivness")
        category_dropdown = gr.Dropdown(label="Select a category:", value="All", choices=categories)
        tone_dropdown = gr.Dropdown(label="Select a tone:", value="All", choices=tones)
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("# Recommended books:")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn = recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()
