import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import  classification_report, log_loss, roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
def season(data):
  '''
    returning the decade from the season
  '''
  return (data%10000)

def fun_inc(data):
  '''
    function to add + 1 seasonID to merge current games records with previous seasons standings

  '''
  return(data+1)

def fun_dec(data):
  '''
    function to dec 1 from SeasonID to get back the original season IDS
  '''
  return(data-1)

def preprocessing():
  '''
    Preprocessing the dataset
  '''
  games_cols = ['GAME_DATE_EST','GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','SEASON','HOME_TEAM_WINS']
  df = pd.read_csv('dataset/games.csv', usecols = games_cols,parse_dates=["GAME_DATE_EST"],infer_datetime_format=True)
  df = df.drop_duplicates().sort_values("GAME_DATE_EST").set_index(["GAME_DATE_EST"])
  team_rank = pd.read_csv('dataset/ranking.csv', parse_dates=['STANDINGSDATE'])
  team_rank.sort_values("STANDINGSDATE",inplace = True)

  team_rank['SEASON_ID'] = team_rank['SEASON_ID'].apply(season)


  team_rank['SEASON_ID'] = team_rank['SEASON_ID'].apply(fun_inc)
  team_rank.drop(["HOME_RECORD","CONFERENCE","LEAGUE_ID","ROAD_RECORD"],axis=1,inplace=True) 
  team_rank.set_index("STANDINGSDATE",inplace=True)

  team_rank.astype({'SEASON_ID': 'int32'})
  df_final_rank = team_rank[team_rank["G"]==82]
  df_final_rank = df_final_rank.drop_duplicates()

  #inner Join for the home team
  new_df = pd.merge(df,df_final_rank.add_suffix("_homeTeam"),how = "inner", left_on=["HOME_TEAM_ID", "SEASON"], right_on=['TEAM_ID_homeTeam', "SEASON_ID_homeTeam"])

  #inner Join for the away team
  new_df = pd.merge(new_df,df_final_rank.add_suffix("_visitorTeam"),how = "inner", left_on=["VISITOR_TEAM_ID", "SEASON"], right_on=['TEAM_ID_visitorTeam', "SEASON_ID_visitorTeam"])
  
  new_df.drop(["TEAM_ID_homeTeam","SEASON_ID_visitorTeam","TEAM_ID_visitorTeam"],axis=1,inplace=True)
  new_df['SEASON_ID_homeTeam'] = new_df['SEASON_ID_homeTeam'].apply(fun_dec)
  return new_df

def fp_fn(conf_matx):
    fp,fn = {},{}
    fn['Win'] = conf_matx[1][0]
    fn['Lose'] = conf_matx[0][1]
    fp['Win'] = conf_matx[0][1]
    fp['Lose'] = conf_matx[1][0]
    return fp,fn

def per_pos(conf_matx):
    ppos = {}
    ppos['Lose'] = conf_matx[0][0]/(conf_matx[0][0]+conf_matx[0][1])
    ppos['Win'] = conf_matx[1][1]/(conf_matx[1][0]+conf_matx[1][1])
    return ppos


if __name__ == '__main__':

  #score list keeps record of all 5 fold accuracies
  score = []
  new_df = preprocessing()
  X = new_df.drop(['SEASON'], axis=1)._get_numeric_data().copy()
  y = X["HOME_TEAM_WINS"]
  X = X.drop(["HOME_TEAM_WINS"],axis=1)
  folds = KFold(n_splits = 5)
  print("total folds = ",folds.get_n_splits(X))
    
    
  for train_ind, test_ind in folds.split(X):
    X_train, X_test = X.iloc[train_ind], X.iloc[test_ind]
    y_train, y_test = y.iloc[train_ind], y.iloc[test_ind]
    
    model=tree.DecisionTreeClassifier(criterion="entropy")

    clf = make_pipeline(StandardScaler(),model)
    clf.fit(X_train,y_train)
    test_preds = clf.predict(X_test)
    print(classification_report(y_true=y_test,y_pred=test_preds))
    score.append(accuracy_score(y_true=y_test,y_pred=test_preds))
    
    
  print(f' ######## 5 Fold Cross Validation Accuracy Score is  :  {sum(score)/len(score)} #####\n\n') 
  conf = confusion_matrix(y_true=y_test,y_pred=test_preds)
  conf_matx = pd.DataFrame(conf, columns=['Predicted loose', 'Predicted Win'],
    index=['Actual loose', 'Actual Win'])

  print(conf_matx)
  
  print('\n \n ### Per Tag Accuracies ####')

  ppos = per_pos(conf)
  print(ppos)

  fp,fn = fp_fn(conf)

  print('\n\n###### False positives #####')
  print(fp)

  print('\n \n#### False Negatives #####')
  print(fn)

  sns.set(font_scale=1.4)
  sns.heatmap(conf,annot=True, annot_kws={"size": 16})
  plt.show()
