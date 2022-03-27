from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = ['aa','aa','bb','cc']
labels = le.fit_transform(labels)
print(labels)