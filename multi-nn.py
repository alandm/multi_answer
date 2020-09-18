import numpy as np
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Model
from IPython.display import SVG
from tensorflow.keras.utils import plot_model

vocabulary_size = 50000
num_income_groups = 10
num_samples= 100
max_length = 10


posts = np.random.randint(1, vocabulary_size, size=(num_samples, max_length))
age_targets = np.random.randint(0, 100, size=(num_samples,1))
income_targets = np.random.randint(1, num_income_groups, size=(num_samples,1))
gender_targets = np.random.randint(0, 2, size=(num_samples,1))


post_input = Input(shape=(None,), dtype = 'int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(post_input)

x = layers.Conv1D(128, 5, activation = 'relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation = 'relu')(x)
x = layers.Conv1D(128, 5, activation = 'relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation = 'relu')(x)
x = layers.Conv1D(128, 5, activation = 'relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation = 'relu')(x)

age_prediction = layers.Dense(1,name='age')(x)
income_prediction = layers.Dense(num_income_groups,activation='softmax',name='income')(x)
gender_prediction = layers.Dense(1,activation='sigmoid', name='gender')(x)


model = Model(post_input, [age_prediction, income_prediction, gender_prediction])

model.compile(optimizer='rmsprop', loss = {'age':'mse',
                                            'income':'categorical_crossentropy',
                                            'gender':'binary_crossentropy'},
                                    loss_weights ={'age':0.25,
                                                    'income':1.,
                                                    'gender': 10.})

print(model.summary())






history = model.fit (posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)


