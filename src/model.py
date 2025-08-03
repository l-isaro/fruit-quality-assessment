from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def build_model(input_shape=(224, 224, 3), num_classes=6, dropout_rate=0.3):
    """
    Builds and compiles a MobileNetV2-based image classification model.

    Args:
        input_shape (tuple): Shape of the input images (H, W, C)
        num_classes (int): Number of output classes
        dropout_rate (float): Dropout rate for regularization

    Returns:
        model (tf.keras.Model): Compiled Keras model
    """
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model for transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def save_model(model, filepath='models/fruit_model.keras'):
    """
    Saves the trained model to a file.
    """
    model.save(filepath)

def load_trained_model(filepath='models/fruit_model.keras'):
    """
    Loads a previously saved model.
    """
    return load_model(filepath)
