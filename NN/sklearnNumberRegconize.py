from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

digits = load_digits()#载入数据
x_data = digits.data #数据
y_data = digits.target #标签

# 标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data) #分割数据1/4为测试数据，3/4为训练数据

mlp = MLPClassifier(hidden_layer_sizes=(100,50) ,max_iter=500)
mlp.fit(x_train,y_train )

predictions = mlp.predict(x_test)
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test,predictions))