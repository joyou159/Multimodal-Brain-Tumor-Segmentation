import tensorflow as tf
import tensorflow.keras.layers as tfl

def Unet_3D_multiclass(width, height, depth, channels, num_classes):

    input = tfl.Input((width, height, depth, channels))

    # contracting path 
    c1  = tfl.Conv3D(filters = 16,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(input)
    c1 = tfl.Dropout(0.1)(c1) 
    c1  = tfl.Conv3D(filters = 16,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c1)
    p1 = tfl.MaxPooling3D(pool_size = (2,2,2))(c1)

    c2  = tfl.Conv3D(filters = 32,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p1)
    c2 = tfl.Dropout(0.1)(c2) 
    c2  = tfl.Conv3D(filters = 32,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c2)
    p2 = tfl.MaxPooling3D(pool_size = (2,2,2))(c2)

    c3  = tfl.Conv3D(filters = 64,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p2)
    c3 = tfl.Dropout(0.2)(c3) 
    c3  = tfl.Conv3D(filters = 64,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c3)
    p3 = tfl.MaxPooling3D(pool_size = (2,2,2))(c3)

    c4  = tfl.Conv3D(filters = 128,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p3)
    c4 = tfl.Dropout(0.2)(c4) 
    c4  = tfl.Conv3D(filters = 128,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c4)
    p4 = tfl.MaxPooling3D(pool_size = (2,2,2))(c4)

    c5  = tfl.Conv3D(filters = 256,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(p4)
    c5 = tfl.Dropout(0.3)(c5) 
    c5  = tfl.Conv3D(filters = 256,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c5)
    u5 = tfl.Conv3DTranspose(filters = 128, kernel_size = (2,2,2), strides = (2,2,2), padding = "same")(c5)

    # expanding path 
    u6 = tfl.concatenate([u5, c4])
    c6  = tfl.Conv3D(filters = 128,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u6)
    c6 = tfl.Dropout(0.2)(c6) 
    c6  = tfl.Conv3D(filters = 128,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c6)
    u6 = tfl.Conv3DTranspose(filters = 64, kernel_size = (2,2,2), strides = (2,2,2), padding = "same")(c6)

    u7 = tfl.concatenate([u6, c3])
    c7  = tfl.Conv3D(filters = 64,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u7)
    c7 = tfl.Dropout(0.2)(c7) 
    c7  = tfl.Conv3D(filters = 64,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c7)
    u7 = tfl.Conv3DTranspose(filters = 32, kernel_size = (2,2,2), strides = (2,2,2), padding = "same")(c7)

    u8 = tfl.concatenate([u7, c2])
    c8  = tfl.Conv3D(filters = 32,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u8)
    c8 = tfl.Dropout(0.1)(c8)
    c8  = tfl.Conv3D(filters = 32,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c8)
    u8 = tfl.Conv3DTranspose(filters = 16, kernel_size = (2,2,2), strides = (2,2,2), padding = "same")(c8)

    u9 = tfl.concatenate([u8, c1])
    c9  = tfl.Conv3D(filters = 16,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(u9)
    c9 = tfl.Dropout(0.1)(c9)
    c9  = tfl.Conv3D(filters = 16,kernel_size = (3,3,3), activation = "relu", kernel_initializer = "he_normal", padding = "same")(c9)

    output = tfl.Conv3D(num_classes, (1,1,1), activation = "softmax")(c9) # multiclass segmentation

    model = tf.keras.Model(inputs = [input], outputs = [output])

    return model  


    

    











    