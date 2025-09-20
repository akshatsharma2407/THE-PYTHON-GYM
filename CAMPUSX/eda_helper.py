import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

def show_full_df(df):
  """
  Function accepts a dataframe
  Returns same dataframe without row and column truncation.
  """
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        display(df)

def null_info(df,ascending=False):
  """
  Accept a dataframe as input
  ascending = [False,True], default is False
  Retuns a dataframe telling the null values count and percentage of each column.
  """
  null_values = df.isna().sum().values
  null_values_pct = (df.isna().sum().values)/df.shape[0]
  return pd.DataFrame({'Columns' : df.columns,'Null_Values' : null_values, 'Null_PCT' : null_values_pct}).set_index('Columns').sort_values(by='Null_Values',ascending=ascending) 

def rotate_xlabels(ax,degree=45):
  """
  Takes ax object as input and rotate the labels to given degree (default is 45)
  """
  return ax.set_xticklabels(
      ax.get_xticklabels(),
      rotation=degree,
      ha='right'
  )

def plot_null_info(data, figsize=(15, 4)):
  """
  plots the Null value count of each column into bar chart.
  by figsize tuple, you can change the size of graph
  """
  na_data = null_info(data)
  fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(15,4),edgecolor="black")

  ax.bar(na_data.index,na_data.Null_Values)
  ax.set_title('Null Value Per Column')
  ax.set_xlabel('Columns')
  ax.set_ylabel('Null Value count')

  rotate_xlabels(ax)
  plt.tight_layout()
  plt.show()

def missingness_corr_plot(data,figsize=(20,10)):
  """
  takes the dataframe as input
  figsize tuple to adjust the graph size
  returns a corelation plot of missingness between each column
  """
  plt.figure(figsize=(20,10))
  corr_df = data.iloc[:,[i for i, n in enumerate(np.var(data.isnull(), axis='rows')) if n > 0]].isna().corr()
  sns.heatmap(corr_df,mask=np.triu(np.ones_like(corr_df,dtype='bool')),linewidth=0.5)
  plt.title('Correlation between Missingness')
  plt.tight_layout()
  plt.show()

def display_html(content,size=2):
  """
  Utility function that takes the content and size
  which display the content in h1,h2....h6 size of HTML
  default is 2
  """
  display(HTML(f"<h{size}>{content}</h{size}>"))

def cat_summary(col):
  """
  Function takes categorical column as input
  Returns a details summary of that column
  """
  display_html(col.name,size=1)

  display_html('Data At A Glance')

  display(col)

  display_html('Meta-Data')
  print(f'Data Type : { col.dtype }')
  print(f'No. of unique value : {col.nunique()}')
  print(f'Missing Rows : {col.isna().sum()} ({col.isna().sum()/col.shape[0]} %)')
  print(f'Available Data : {col.notna().sum()} / {col.shape[0]} rows')

  display_html('Description/Summary')
  display(col.describe())

  display_html('Value count & PCT')
  display(show_full_df(pd.concat([col.value_counts(),(col.value_counts(normalize=True)*100).round(3)],axis=1)))

def categorical_univariate_plot(col,k=-1,wrdcld=True):
  """
  Function takes categorical column as input
  k is used to see only top k categories, default is all categories (-1)
  Returns a bar, pie and wordcloud (if wrdcld set to True)
  """
  fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))

  vc = col.value_counts()

  if k == -1:
    data = vc
  else:
    val = (vc).iloc[k:].values.sum()
    data = pd.concat([vc[:k],(pd.Series(vc[k:].sum(), index=["Others"]))]).sort_values(ascending=False)

  ex = [0.1] + [0 for i in range((data.index).nunique()-1)]
  ax[0].pie(data,autopct='%0.01f%%',labels=data.index,explode=ex,shadow=True,startangle=90)

  ax[1].bar(data.index,data.values)
  rotate_xlabels(ax[1])

  fig.suptitle(f'Distribution of Category in {col.name}')
  fig.tight_layout()

  plt.show()

  if wrdcld:
    text = " ".join(col.dropna().to_list())
    
    plt.figure(figsize=(15,10))
    word_cloud = WordCloud().generate(text)

    plt.imshow(word_cloud)

    plt.show()
