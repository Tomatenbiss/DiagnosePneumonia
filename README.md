# Diagnose Pneumonia

## Original Setting

image_height = 150
image_width = 150
batch_size = 32
no_of_epochs  = 300

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

no of epochs:   2       5      10     15    20      25      30      35      40      45      50      55
loss:           .4453   .3881  .3348  .2744 .295    .3088   .2872   .2996   .2926   .2977   .3013   .3017
accuracy:       .8207   .8355  .8898  .9112 .9145   .9079   .9145   .9128   .9128   .9128   .9128   .9112

no of epochs:   60      65      70      75      80      85      90      95      100     105     110     115
loss:           .2986   .2950   .2956   .2944   .3065   .2883   .2978   .3008   .3094   .2961   .3028   .3009
accuracy:       .9145   .9112   .9128   .9112   .9095   .9145   .9112   .9112   .9062   .9128   .9095   .9112

no of epochs:   120     125     130     135     140     145     150     155     160     165     170     175
loss:           .2984   .2990   .3017   .2977   .3002   .3037   .2940   .2977   .3095   .3011   .2983   .2886
accuracy:       .9128   .9128   .9145   .9194   .9145   .9145   .9194   .9178   .9112   .9128   .9194   .9211

no of epochs:   180     185     190     195     200     205     210     215     220     225     230     235
loss:           .3052   .2976   .3159   .31     .2992   .3136   .2987   .3056   .3108   .3063   .2892   
accuracy:       .9178   .9211   .9128   .9128   .9178   .9128   .9145   .9145   .9128   .9194   .9227

### Setting with dropout layers reduced to one with a dropout rate

image_height = 150
image_width = 150
batch_size = 32
no_of_epochs  = 300

### Paper implementation with no dropout

image_height = 32
image_width = 32
batch_size = 32
no_of_epochs  = 300

model = Sequential()
model.add(Conv2D(16,(2,2),input_shape=(image_height,image_width,3),activation='linear'))
model.add(LeakyReLU(alpha=.3))
model.add(Conv2D(36,(2,2),activation='linear'))
model.add(LeakyReLU(alpha=.3))
model.add(Conv2D(64,(2,2),activation='linear'))
model.add(LeakyReLU(alpha=.3))
model.add(Conv2D(100,(2,2),activation='linear'))
model.add(LeakyReLU(alpha=.3))
model.add(Conv2D(144,(2,2),activation='linear'))
model.add(LeakyReLU(alpha=.3))
model.add(AveragePooling2D(pool_size=(27,27)))
model.add(Flatten())
model.add(Dense(units=864,activation='relu'))
model.add(Dense(units=288,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### Paper implementation with dropout (.5) before first Dense layer

