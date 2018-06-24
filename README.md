# Diagnose Pneumonia

## Original Setting

```
model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(image_height,image_width,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Results
image_height = 150\
image_width = 150\
batch_size = 32\
no_of_epochs  = 2

**loss**: .357\
**accuracy**: .8487

image_height = 150\
image_width = 150\
batch_size = 32\
no_of_epochs  = 5

**loss**: .2936\
**accuracy**: .8914