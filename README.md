# Store-Sales---Time-Series-Forecasting
![image](https://github.com/user-attachments/assets/4a504b1b-9b03-41ad-972d-a5b2e5fadd29)

## Description
In this “getting started” competition, you’ll use time-series forecasting to forecast store sales on data from Corporación Favorita, a large Ecuadorian-based grocery retailer.
Specifically, you'll build a model that more accurately predicts the unit sales for thousands of items sold at different Favorita stores. You'll practice your machine learning skills with an approachable training dataset of dates, store, and item information, promotions, and unit sales.

__________________________________________________________

В этом конкурсе "Начало работы" вы будете использовать прогнозирование временных рядов для прогнозирования продаж в магазинах на основе данных, полученных от Corporación Favorita, крупной эквадорской компании, занимающейся розничной торговлей продуктами питания.
В частности, вы построите модель, которая будет более точно предсказывать удельные продажи для тысяч товаров, продаваемых в разных магазинах Favorita. Вы отработаете навыки машинного обучения на доступном обучающем наборе данных, содержащем информацию о датах, магазинах, товарах, акциях и продажах.

## Score
RMSLE: 0.47574813032260616
MAE: 60.13507020460128
R2: 0.9518265728848506

<pre>
```
import matplotlib.pyplot as plt
import seaborn as sns

# Вычисляем корреляцию только с целевой переменной 'sales'
correlation_with_sales = X_train.corr(numeric_only=True)[['sales']].sort_values(by='sales', ascending=False)

# Создаём тепловую карту
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_with_sales, annot=True, cmap='coolwarm', cbar=True, fmt='.2f')
plt.title('Correlation with Sales')
plt.show()
```</pre>
![image](https://github.com/user-attachments/assets/4aed6966-e3fc-4c1b-bdd6-7e4279e0fc03)

## Pipeline
<pre>
```
# Предобработка данных
numerical_pipeline = Pipeline([
    ('fillna', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


categorical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encoding', OneHotEncoder(handle_unknown='ignore', sparse=True))
])


preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),  # Для числовых признаков
    ('cat', categorical_pipeline, categorical_features)  # Для категориальных
    ], 
    remainder='passthrough'
)


# Модель 
model = xgb.XGBRegressor(
    objective="reg:tweedie",  # Используем правильное значение для Tweedie Loss
    n_estimators=987,  # Количество деревьев
    learning_rate=0.03974568823588938,  # Скорость обучения
    depth=14,  # Максимальная глубина дерева
    min_data_in_leaf=5,  # Минимальное количество данных в узле (аналог min_child_weight)
    subsample=0.7,  # Доля выборки для построения дерева
    colsample_bylevel=0.5,  # Доля признаков на уровне дерева
    reg_lambda=0.3882731379185479,  # L2-регуляризация
    reg_alpha=0.009556952677129724  # L1-регуляризация
)



# Создаем финальный пайплайн с выбранной моделью
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])
```
</pre>
