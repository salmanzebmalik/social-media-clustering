import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def scatter_2d(X, labels, title="2D Projection"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=labels, palette="tab10", s=30, alpha=0.8)
    plt.title(title)
    plt.show()

def wordcloud_from_tokens(tokens, title="WordCloud"):
    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
