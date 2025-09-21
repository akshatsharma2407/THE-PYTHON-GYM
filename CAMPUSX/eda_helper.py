import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from scipy import stats
import warnings

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


def get_top_k(vc,k):
  """
  In value_counts(), it retains the top k categories
  and make all other as "others"
  """
  val = (vc).iloc[k:].values.sum()
  return pd.concat([vc[:k],(pd.Series(vc[k:].sum(), index=["Others"]))]).sort_values(ascending=False)

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
    data = get_top_k(vc,k)

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

def correlation_heatmap_numerical(df,exclude=[],method='spearman',annot=True,linewidth=0.1,figsize=(15,5)):
  """
  Function expect a dataframe

  Returns a correlation heatmap of numerical columns ('float','int') data types columns

  Parameters ->
  1. Exclude - list of columns that you don't want to show in figure.
  2. method - default is spearman. can hold value of -> spearman,pearson.
  3. annot - default is True, it can hold value of True or False.
  4. linewidth - default is 0.1, for better visual, set to 0, if you don't need it.
  5. figsize - to adjust the size of graph
  """
  corr_matrix = (df.select_dtypes(include=['float','int']).drop(columns=exclude).corr(method=method)).round(1)
  plt.figure(figsize=figsize)
  mask = np.triu(np.ones_like(corr_matrix,dtype='bool'))
  sns.heatmap(corr_matrix,mask=mask,linewidth=linewidth,annot=annot)
  plt.title('Numerical Columns correlation matrix')
  plt.show()

def num_cat_bivariate_plots(df,col1,col2,bar_estimator=np.mean,bar_xticks_rotation=45,FacetGrid_col_wrap=3,bar_plot_fig_size=(15,5)):
  """
  Function returns 3 plot showing the relation between Categorical and Numerical Column

  Parameters : 
  1. col1 - Categorical Column name.
  2. col2 - Numerical Column name.
  3. bar_estimator - [np.mean,np.median], Default is np.mean, helpful to determine the bar height
  4. bar_xticks_rotation - to rotate Bar plot's X-Axis labels.
  5. FacetGrid_col_wrap - to determine the num of plot in a column in box and violin plot, Default is 3
  6. bar_plot_fig_size - to adjust the size of bar chart
  """
  display_html(f'{col1} Vs {col2} Bivariate Analysis Plots',size=1)
  print('\n\n\n\n')
  
  warnings.filterwarnings('ignore')

  display_html(f'{col1} Vs {col2} Bar Chart with {str(bar_estimator.__name__)} Estimator')
  plt.figure(figsize=(bar_plot_fig_size))
  sns.barplot(data=df,x=col1,y=col2,estimator=bar_estimator)
  plt.xticks(rotation=bar_xticks_rotation,ha='right')
  plt.tight_layout()
  plt.show()

  print('\n\n\n\n\n')

  display_html(f'Box Plot of {col1} And {col2}')
  g = sns.FacetGrid(data=df,col=col1,col_wrap=FacetGrid_col_wrap,sharex=False)
  g.map(sns.boxplot,col2)
  plt.tight_layout()
  plt.show()

  print('\n\n\n\n\n')

  display_html(f'Violin Plot Between {col1} and {col2}')
  g = sns.FacetGrid(data=df,col=col1,col_wrap=FacetGrid_col_wrap,sharex=False)
  g.map(sns.violinplot,col2)
  plt.tight_layout()
  plt.show()


def numerical_summary(col):
  """
  Function provide Numerical Summary
  1. Data At a Glance.
  2. Meta Data.
  3. Central Tendency
  4. Spread of Data
  5. Kurtosis & Varience
  """
  display_html('Data At a Glance',size=1)
  display(col)

  display_html('Meta Data')
  print(f'Data Type : {col.dtype}')
  print(f'Missing Data : {col.isna().sum()} ({round((col.isna().sum()/col.shape[0])*100,2)} %)')
  print(f'Available Data : {col.notna().sum()} / {col.shape[0]}')

  display_html('Percentiles')
  display(col.quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,1]))

  display_html('Central Tendency')
  display(pd.Series(
      {
          'Mean' : round(col.mean(),2),
          'Trimmed Mean (5%)' : round(stats.trim_mean(col,proportiontocut=0.05),2),
          'Trimmed Mean (10%)' : round(stats.trim_mean(col,proportiontocut=0.1),2),
          'Median' : col.median()
      }
  ).rename('value'))

  display_html('Spread of Data')
  display(pd.Series(
      {
          'Std' : col.std(),
          'Varience' : col.var(),
          'IQR' : col.quantile(0.75) - col.quantile(0.25)
      }
  ).rename('value').to_frame().style.format("{:,.0f}"))

  display_html('Kurtosis & Varience')
  display(pd.Series(
      {
          'Skewness' : col.skew(),
          'Kurtosis' : col.kurtosis()
      }
  ).rename('value').to_frame().style.format("{:,.0f}"))

def numerical_univariate_plots(col,hist_bins='auto',figsize=(20,10),power_transform_method='yeo-johnson',qqplot_dist=stats.norm):
  """
  create figures related to univariate analysis of Numerical column

  parameter ->
  1. col - numerical column name.
  2. hist_bins - for custom histogram binning, default is auto.
  3. figsize - adjust the figure.
  4. power_transform_method - ['Yeo-Johnson','Box-Cox'], default is 'Yeo-Johnson'
  5. qqplot - can take any continuous distribution object from scipy.stats, like norm, lognormal, t etc.
  """
  fig,ax = plt.subplots(nrows=2,ncols=3,figsize=figsize)

  colors_seq = sns.color_palette("RdYlBu", 6)

  sns.histplot(col,bins=hist_bins,kde=True,ax=ax[0][0],color=colors_seq[0])
  ax[0][0].set_title('Histogram')

  sns.ecdfplot(col,ax=ax[0][1],color=colors_seq[1])
  ax[0][1].set_title('CDF')

  sns.boxplot(col,ax=ax[1][0],color=colors_seq[2])
  ax[1][0].set_title('Box-Plot')

  sns.violinplot(col,ax=ax[1][1],color=colors_seq[3])
  ax[1][1].set_title('Violin-Plot')

  sns.histplot(PowerTransformer(method=power_transform_method).fit_transform(col.values.reshape(-1,1)),ax=ax[0][2],color=colors_seq[4])
  ax[0][2].set_title(power_transform_method)

  sm.qqplot(data=col.dropna(),line='45',dist=qqplot_dist,fit=True,ax=ax[1][2],color=colors_seq[5])
  ax[1][2].set_title('QQ-Plot')

  fig.suptitle('Numerical Univariate Plots',fontsize=20)
  fig.tight_layout()
  fig.show()
