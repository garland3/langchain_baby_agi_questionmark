
import os

data = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test = train_test_split(data.drop('temperature', axis=1),
                                                    data['temperature'],
                                                    test_size=0.2,
                                                    random_state=0)
parameters = {'max_depth': [2, 4, 6, 8],
              'min_samples_leaf': [2, 4, 6, 8],
              'min_samples_split': [2, 4, 6, 8]}
clf = GridSearchCV(DecisionTreeRegressor(), parameters, cv=5)
clf.fit(X_train, y_train)
model = DecisionTreeRegressor(max_depth=clf.best_params_['max_depth'],
                              min_samples_leaf=clf.best_params_['min_samples_leaf'],
                              min_samples_split=clf.best_params_['min_samples_split'])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

features = list(data.columns.drop('temperature'))
summary = f"The data used for the model consists of {features} and the target variable is temperature."
print(summary)
print(f"The model achieved an accuracy of {accuracy} on the test set.")
os.system('echo "Todo List: \n- Tune additional model parameters \n- Try different model architectures \n- Try different hyperparameter optimization techniques."')