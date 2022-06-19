tsne = TSNE(n_components=2, perplexity=2)
df_transposed = dft.T
# print(df_transposed.head())
tsne_results = tsne.fit_transform(df_transposed)
print(tsne_results.shape)
print(tsne_results)

dd = pd.DataFrame(tsne_results)

# X,y = make_classification(n_samples=10, n_features=5, n_informative=5, n_redundant=0, random_state=0, n_classes=3)
X, y = make_multilabel_classification(n_samples=10, n_features=7, n_classes=5, n_labels=2)
print(X, y)
plt.plot(y)
plt.show()
sys.exit()

# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)

# define the model
model = RandomForestClassifier()

# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [
    [-8.52381793, 5.24451077, -12.14967704, -2.92949242, 0.99314133, 0.67326595, -0.38657932, 1.27955683, -0.60712621,
     3.20807316, 0.60504151, -1.38706415, 8.92444588, -7.43027595, -2.33653219, 1.10358169, 0.21547782, 1.05057966,
     0.6975331, 0.26076035]]
yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])