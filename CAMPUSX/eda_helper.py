import pandas as pd
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from scipy import stats
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def rotate_ylabels(ax,degree=45):
  """
  Takes ax object as input and rotate the labels to given degree (default is 45)
  """
  return ax.set_yticklabels(
      ax.get_yticklabels(),
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
  print(f'Missing Rows : {col.isna().sum()} ({round((col.isna().sum()/col.shape[0])*100,2)} %)')
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

  print('\n'*2)

  if wrdcld:
    text = " ".join(col.dropna().astype(str).to_list())
    plt.figure(figsize=(7,5))
    word_cloud = WordCloud(background_color="white").generate(text)

    plt.imshow(word_cloud)
    plt.axis("off")
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

  for row in ax:
      for a in row:
          a.yaxis.set_major_formatter(mticker.EngFormatter())  # formats 1e6 → 1M
          a.xaxis.set_major_formatter(mticker.EngFormatter())
  fig.show()

def categorical_bivariate_plots(df,col1,col2,figsize=(15,5),normalize='index',rotate_x_axis=45,rotate_y_axis=0,stacked=True):
  """
  Returns Graphs showing relation between two categorial variables

  parameters ->
  1. df - dataframe
  2. col1, col2 - name of categorical columns
  3. figsize - to adjust the figure size.
  4. normalize - [Ture,'index','columns'] Default is 'index'.
  5. rotate_x_axis - rotate lables of x_axis on heatmap.
  6. rotate_y_axis - rotate labels of y_axis on heatmap.
  7. stacked - [True,False] Default is True. returns grouped bar chart if set to False
  """
  display_html(f'{col1} Vs {col2} Categorical Plots',size=1)

  fig,ax = plt.subplots(nrows=1,ncols=2,figsize=figsize)

  crsstb = pd.crosstab(index=df[col1],columns=df[col2])
  crssstb_nc = pd.crosstab(index=df[col1],columns=df[col2],normalize=normalize).round(2)

  sns.heatmap(crsstb,linewidths=1.5,annot=True,fmt="d",cmap='Oranges',ax=ax[0])
  ax[0].set_title('Cross Tab')
  rotate_xlabels(ax[0],degree=rotate_x_axis)
  rotate_ylabels(ax[0],degree=rotate_y_axis)

  sns.heatmap(crssstb_nc,linewidths=1.5,annot=True,cmap='Greens',ax=ax[1])
  ax[1].set_title('Cross Tab - Normalized')
  rotate_xlabels(ax[1],degree=rotate_x_axis)
  rotate_ylabels(ax[1],degree=rotate_y_axis)

  fig,ax = plt.subplots(figsize=figsize)
  crsstb.plot(kind='bar',ax=ax,stacked=stacked)
  ax.legend(loc="upper left",bbox_to_anchor=(1,1))
  ax.set_title('Bar-Plot',fontsize=20)

  g = sns.clustermap(crsstb, cmap="coolwarm", annot=True, fmt="d",figsize=figsize)
  g.fig.suptitle('Clusermap',fontsize=20)
  plt.tight_layout()

  fig.tight_layout()
  fig.show() 

def numerical_bivariate_plot(df,x,y,joint_plot=sns.scatterplot,margin_plot=sns.histplot,method='spearman'):
  """
  Functions plots two graph, 1. joinplot 2. correlation plot
  Parameters ->
  1. joint_plot - Default is sns.scatterplot, Here we can pass any bivariate numerical plot object of seaborn.
  2. margin_plot -Default is sns.histplot, Any Object related to univariate numerical plot object of seaborn.
  3. method - Default is spearman, avail options are ['spearman','pearson'].
  """
  g = sns.JointGrid(data=df,x=x,y=y)
  g.plot(joint_plot,margin_plot)
  g.figure.suptitle('Joint-Grid')
  plt.tight_layout()
  plt.show()

  print('\n'*2)

  fig,ax = plt.subplots(nrows=1,ncols=1)
  corr_matrix = df[[x,y]].corr(method=method)
  sns.heatmap(corr_matrix,ax=ax,annot=True,linewidth=1.5,cmap='Greens')
  plt.title('Correlation Matrix')
  plt.tight_layout()
  plt.show()

def numerical_vs_numerical_hypothesis_testing(df,var1,var2,alpha=0.05):
  """
  Function does two Test 
  1. Pearson 2. Spearsman test

  df,var1,var2 - Dataframe, name of col1, name of col2
  alpha - Significance level, decides whether to accept or reject the null hypothesis
  """

  display_html('Pearson Correlation Test')
  test_df = df[[var1,var2]].dropna()

  statistic,pvalue = stats.pearsonr(test_df[var1],test_df[var2])

  print(f'Significance level : {0.05*100} %')
  print('Null Hypothesis : The Samples are not correlated.')
  print('Alternate Hypothesis : The Samples are correlated.')
  print(f'Test Statistic : {statistic}')
  print(f'Pvalue : {pvalue}')

  if pvalue < alpha:
    print(f'Since Pvalue is less than {alpha}, we reject the Null Hypothesis.')
    print(f'CONCLUSION: The Columns {var1} and {var2} are correlated.')
  else:
    print(f'Since Pvalue is more than {alpha}, we Failed to reject the Null Hypothesis.')
    print(f'CONCLUTSION: The Columns {var1} and {var2} are not correlated.')
  
  display_html('Spearman Correlation Test')

  statistic,pvalue = stats.spearmanr(test_df[var1],test_df[var2])

  print(f'Significance level : {0.05*100} %')
  print('Null Hypothesis : The Samples are not correlated.')
  print('Alternate Hypothesis : The Samples are correlated.')
  print(f'Test Statistic : {statistic}')
  print(f'Pvalue : {pvalue}')

  if pvalue < alpha:
    print(f'Since Pvalue is less than {alpha}, we reject the Null Hypothesis.')
    print(f'CONCLUSION: The Columns {var1} and {var2} are correlated.')
  else:
    print(f'Since Pvalue is more than {alpha}, we Failed to reject the Null Hypothesis.')
    print(f'CONCLUTSION: The Columns {var1} and {var2} are not correlated.')


def shapiro_test(col, alpha=0.05):
    """
    Function checks for normality by performing shapiro test based on given alpha value (default is 0.05)
    and provides the  conclusion.

    Parameter : 
    1. col - column on which shapiro test will be applied
    2. alpha - Default is 0.05. significance level, decides whether to accept or reject the null hypotheses
    """
    warnings.filterwarnings('ignore')

    statistic, pvalue = stats.shapiro(col.dropna())

    print(f'Significance level : {alpha*100} %')
    print('Null Hypothesis : The samples are Normally Distributed.')
    print('Alternate Hypothesis : The samples are Not Normally Distributed.')
    print(f'Test Statistic : {statistic}')
    print(f'Pvalue : {pvalue}')

    if pvalue < alpha:
        print(f'Since Pvalue is less than {alpha}, we reject the Null Hypothesis.')
        print(f'CONCLUSION: The column "{col.name}" is Not Normally Distributed.')
        
    else:
        print(f'Since Pvalue is greater than {alpha}, we fail to reject the Null Hypothesis.')
        print(f'CONCLUSION: The column "{col.name}" can be considered Normally Distributed.')
        


def levene_test(df,numeric_col,category_col,alpha=0.05):
  """
  Function to test equal variances (homoscedesticity)
  Parameter ->
  1. groups - different groups on which test need to be applied, this groups should be array.
  2. significance_level - Default is 0.05, used to determine whether to accept or reject null hypothesis.
  3. Returns - [True,False] which will be used if other function calls.
  """

  test_df = df[[numeric_col,category_col]].dropna()

  groups_dict = {name:group[numeric_col].values for name,group in test_df.groupby(category_col)}

  statistic,pvalue = stats.levene(*groups_dict.values())
  
  print(f'Significance level : {alpha*100} %')
  print(f'Null Hypothesis : All Groups have equal variances.')
  print(f'Alternate Hypothesis : At least one group has different variance.')
  print(f'Test Statistic : {statistic}')
  print(f'Pvalue : {pvalue}')

  if pvalue < alpha:
    print(f'Since Pvalue is less than {alpha}, we reject Null Hypothesis')
    print('CONCLUSION : Variances between groups are not equal.')
    return False
  
  else:
    print(f'Since Pvalue is greater than {alpha}, we Failed to reject Null Hypothesis')
    print('CONCLUSION : Variances between groups are equal.')
    return True

def numerical_vs_categorical_hypothesis_test(df,numeric_col,category_col,comparison_name=None,anova_alpha=0.05,levene_alpha=0.05,post_hoc_alpha=0.05):
  """
  Perform levene,one_way_anova & Post-hoc Test.
  
  Parameters ->
  1. df - The DataFrame Object.
  2. Numeric_col - Name of Numerical column
  3. Category_col - Name of Categorical Column which will act as groups for levene and anova test.
  4. comparison_name - Name of group (category within category col), will be used as parameter in Post-hoc Plot.
  5. anova_alpha, levene alpha, post_hoc_alpha - values that will decide respective Test result.
  """
  display_html('Levene Test')
  print('----------------')

  result = levene_test(df,numeric_col,category_col,alpha=levene_alpha)

  if not result:
    display_html(f' "{numeric_col}" with "{category_col}" groups does not pass levene test. Hence we will use Welch’s ANOVA test',5)
  
  display_html('One Way Anova')
  print('---------------------')

  test_df = df[[numeric_col,category_col]].dropna()
  groups_dict = {name:group[numeric_col].values for name,group in test_df.groupby(category_col)}

  statistic,pvalue = stats.f_oneway(*groups_dict.values(),equal_var=result)

  print(f'Significance level : {anova_alpha*100}%')
  print(f'Null Hypothesis : All Group means are equal.')
  print(f'Alternate Hypothesis : One or more Group mean is different')

  if pvalue < anova_alpha:
    print(f'Since Pvalue is less than {anova_alpha}, we reject Null Hypothesis')
    print('CONCLUSION : Atleast one group mean is different')
  else:
    print(f'Since Pvalue is more than {anova_alpha}, we Failed to reject Null Hypothesis')
    print('CONCLUSION : All group means are Equal')

  display_html(f'Post-Hoc Test (Tukey-HSD)')
  print('--------------------------------------\n')
  tukey = pairwise_tukeyhsd(endog=test_df[numeric_col],groups=test_df[category_col],alpha=post_hoc_alpha)

  print(tukey.summary())
  print('\n')
  tukey.plot_simultaneous(comparison_name=comparison_name)

  plt.show()

def categorical_categorical_hypothesis_test(df,col1,col2,alpha=0.05,show_freq=False):
  """
  This function perform chi2 (Test for Independence), between two categorical variables

  1. df - DataFrame object
  2. col1,col2 - column name on which test will be applied
  3. alpha - Default value of alpha is 0.05, It is used to accept or reject the null hypothesis
  4. show_freq - Whether to show or hide the Observed and Expected Frequencies.
  """

  ct = pd.crosstab(index=df[col1],columns=df[col2])
  statistic,pvalue,dof,exp_freq = stats.chi2_contingency(ct)

  if show_freq:
    display_html('Observed Frequency')
    print('----------------------------')
    display(ct)

    display_html('Expected Frequency')
    print('---------------------------')
    display(pd.DataFrame(exp_freq).set_axis(ct.columns,axis=1).set_axis(ct.index,axis=0))

  display_html('Chi-Square Test for Independence')
  print('---------------------------------------------')
  print(f'Significance level : {alpha*100} %')
  print(f'Null Hypotheses : There is no association between two categorical variables.')
  print(f'Alternate Hypotheses : There is association between two categorical variables.')
  print(f'Degree of Freedom : {dof}')

  if pvalue < alpha:
    print(f'Since Pvalue is less than {alpha}, we reject Null Hypothesis')
    print(f'CONCLUSION :  There is association between {col1} and {col2}')
  else:
    print(f'Since Pvalue is more than {alpha}, we failed to reject Null Hypothesis')
    print(f'CONCLUSION : There is no association between {col1} and {col2}')  

def boxplot_outlier(df,column,boxplot=True):
  q1 = df[column].quantile(0.25)
  q3 = df[column].quantile(0.75)
  iqr = q3 - q1
  min = q1 - (1.5*iqr)
  max = q3 + (1.5*iqr)

  if boxplot:
    fig,ax = plt.subplots(ncols=2,figsize=(15,5))
    sns.boxplot(data=df,x=column,ax=ax[0])
    ax[0].set_title('Before IQR Outlier Removal')
    sns.boxplot(data = df.query(f"{column} > {min} & {column} < {max}"),x=column,ax=ax[1])
    ax[1].set_title('Afer IQR Outlier Removal')
    plt.show()

  return df.query(f"{column} <= {min} | {column} >= {max}")
