

def train_tree_model(data):   
    # group: s1

    lstree =[]

    group1 = data.groupby('model')
#ls_prophet_prediction= []
    for g1 in group1.groups:
        groupm = group1.get_group(g1)
        shape = groupm.shape[0]
    #print(len(groupm['model']))
        if shape <10:
            pass
        else:
            print('start training model:',groupm['model_code'].unique())
            print('--------------------continue-----------------------')
            numerical = groupm.drop(['weekid','model','date','model_code'], axis=1)

    
    #df_cat=  groupm[['weekid', 'model_code','date']]
    
            length_df = groupm.shape[0]
            df_train = groupm.iloc[:length_df - 4]
            df_val = groupm.iloc[len(groupm)-4:]
    
            label = numerical.pop('sum_qty')
            features = numerical
            selected = feature_selection(features,label,numerical)
#     print(selected)
      
            if selected == []:
                #print('using all features')
                #print('shape of data is:',shape)
                data = numerical
                train_df = df_train
                val_df = df_val
        
            else:
                #print('selected features are:',selected)
                print('shape of data is:',shape)
        #df_num = groupm[selected]
        #sel = selected
        #print('selected feature are:',selected)
                target = 'sum_qty'
                selected.append(target)
                train_df = df_train[selected]
                val_df = df_val[selected]
        
        Y_train = train_df.pop('sum_qty')
        X_train = train_df
        Y_valid = val_df.pop('sum_qty')
        X_valid = val_df
    
        rf =  RandomForestRegressor().fit(X_train,Y_train)
        xgb = XGBRegressor().fit(X_train,Y_train)
        etree = ExtraTreesRegressor().fit(X_train,Y_train)
        gboost = GradientBoostingRegressor().fit(X_train,Y_train)
        lgbm = LGBMRegressor().fit(X_train,Y_train)
        dt = DecisionTreeRegressor().fit(X_train,Y_train)
        knn = KNeighborsRegressor().fit(X_train,Y_train)
        ada = AdaBoostRegressor().fit(X_train,Y_train)
        cat = CatBoostRegressor().fit(X_train,Y_train,silent=True)
    
    ################################################################
   
        yhat_rf = rf.predict(X_valid)
        yhat_xgb = xgb.predict(X_valid)
        yhat_etree = etree.predict(X_valid)
        yhat_gboost = gboost.predict(X_valid)
        yhat_lgbm = lgbm.predict(X_valid)
        yhat_dt = dt.predict(X_valid)
        yhat_knn = knn.predict(X_valid)
        yhat_ada = ada.predict(X_valid)
        yhat_cat = cat.predict(X_valid)
   
    ################################################################
    
        d1 = {'week':df_val['weekid'].values,
          'model':df_val['model_code'].values,
          'y':df_val['sum_qty'].values,
          'yhat_rf':yhat_rf,
          'yhat_xgb':yhat_xgb,
          'yhat_etree':yhat_etree,
          'yhat_gboost':yhat_gboost,
          'yhat_lgbm':yhat_lgbm,
                    'yhat_dt':yhat_dt,
                    'yhat_knn':yhat_knn,
                    'yhat_ada':yhat_ada,
                    'yhat_cat':yhat_cat
        }
    ################################################################
        gap= pd.DataFrame(d1)
        gap = gap.groupby(['model']).agg({'y': 'sum',  'yhat_rf': 'sum','yhat_xgb': 'sum',    
                                           'yhat_etree': 'sum',
                                           'yhat_gboost': 'sum','yhat_lgbm': 'sum',
     'yhat_dt': 'sum','yhat_knn': 'sum','yhat_ada': 'sum','yhat_cat': 'sum'
                                          }).reset_index()    
    
        gap['gap_rf'] = abs(gap['y']-gap['yhat_rf']) / gap['yhat_rf']
        gap['gap_xgb'] = abs(gap['y']-gap['yhat_xgb']) / gap['yhat_xgb']
        gap['gap_etree'] = abs(gap['y']-gap['yhat_etree']) / gap['yhat_etree']
        gap['gap_gboost'] = abs(gap['y']-gap['yhat_gboost']) / gap['yhat_gboost']
        gap['gap_lgbm'] = abs(gap['y']-gap['yhat_lgbm']) / gap['yhat_lgbm']
        gap['gap_dt'] = abs(gap['y']-gap['yhat_dt']) / gap['yhat_dt']
        gap['gap_knn'] = abs(gap['y']-gap['yhat_knn']) / gap['yhat_knn']
        gap['gap_ada'] = abs(gap['y']-gap['yhat_ada']) / gap['yhat_ada']
        gap['gap_cat'] = abs(gap['y']-gap['yhat_cat']) / gap['yhat_cat']
    
        gap['gap_best']= gap[['gap_rf','gap_xgb', 
          'gap_etree','gap_gboost','gap_lgbm','gap_dt',
        'gap_knn','gap_ada','gap_cat']].min(axis=1)
    
        gap['prediction'] = gap['y'] / (1-gap['gap_best'])
    
        gap = gap[['model','y','prediction','gap_best']]
    ###############################################################
        lstree.append(gap)
    return lstree


treemodel = train_tree_model(data)
output = pd.concat(treemodel)


def train_lstm_model(data,n_input,n_out,epochs,batch):  
    ls_rnn = []
    group1 = data.groupby('model')
#ls_prophet_prediction= []
    for g1 in group1.groups:
        groupm = group1.get_group(g1)
        shape = groupm.shape[0]
        print('start training model:',groupm['model'].unique())
        print('length of the model is:',shape)       
        numerical = groupm.drop(['weekid','model','date','model_code'], axis=1)
        length_df = groupm.shape[0]
        df_train = groupm.iloc[:length_df - 4]
        df_val = groupm.iloc[len(groupm)-4:]
    
        label = numerical.pop('sum_qty')
        features = numerical
        selected = feature_selection(features,label,numerical)
        
        if selected == []:
            data = numerical
            train = normal(df_train)
            valid = normal(df_val)
        
        else:
            target = 'sum_qty'
            selected.append(target)    
            train= normal(df_train[selected])
            valid = normal(df_val[selected]) 
#     print(selected)
    train = array(split(train, len(train)/1))
    valid = array(split(valid, len(valid)/1))    
    #print(train.shape,valid.shape,numerical.shape)
################################################################################################### 
        x_train,y_train = to_supervised(train, n_input=n_input,n_out=n_out) # 14 wks to predict 7 wks
        x_test, y_test = to_supervised(valid, n_input=n_input,n_out=n_out)
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
	# reshape output into [samples, timesteps, features]
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))    
        model = Sequential()
        model.add(LSTM(200, activation='tanh', input_shape=(n_timesteps, n_features)))
        model.add(Dropout(rate=0.2))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),loss='mean_absolute_percentage_error',metrics=['mse', 'mape']) 
        print('------------------------------------------------------------')
        model.fit(x_train, y_train, epochs=epochs,batch_size=batch, shuffle=False,verbose=0)
    
    
    #####################################################################
        valid_predict = model.predict(valid)
        valid_predict = valid_predict.T[0] #3,30  
        valid1 = valid_predict[0].reshape(-1,1) #157
    ####################################################
        target = df_val[['sum_qty']]
        scaler = StandardScaler()
        target_scaled = scaler.fit_transform(target)
    

        y = df_val.pop('sum_qty').tolist()
        yhat = pd.DataFrame(scaler.inverse_transform(valid1))
        yhat.columns=['yhat_lstm']


        d2 = {'week':df_val['weekid'].values,
          'model':df_val['model'].values,
          'y':target['sum_qty'].values,
          'yhat_lstm':yhat['yhat_lstm'].values}
    
        gap = pd.DataFrame(d2)
        gap = gap.groupby(['model']).agg({'y': 'sum', 
                                                      'yhat_lstm': 'sum'
                                          }).reset_index()
        gap['gap_lstm'] = abs(gap['yhat_lstm'] - gap['y'])/ gap['yhat_lstm']
        ls_rnn.append(gap)
    
    return ls_rnn  

lstm = train_tree_model(data,n_input=1,n_out=1,epochs=500,batch=16)
output1 = pd.concat(lstm)






